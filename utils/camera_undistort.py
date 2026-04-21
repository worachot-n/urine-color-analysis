"""
Modular camera undistortion pipeline for Raspberry Pi Camera Module V3.

Usage — dataset collection (saves undistorted frames):
    python -m utils.camera_undistort --mode collect --output /tmp/frames

Usage — real-time preview:
    python -m utils.camera_undistort --mode preview

Usage — as a library:
    from utils.camera_undistort import CameraUndistorter
    u = CameraUndistorter("configs/camera_params.yaml", capture_size=(1920, 1080))
    frame_bgr = ...                        # captured from Picamera2
    clean     = u.undistort_frame(frame_bgr)
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

# Picamera2 is only available on ARM Linux (Raspberry Pi).
try:
    from picamera2 import Picamera2
    from libcamera import controls as _lc_controls
    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class CalibrationParams:
    """Parsed contents of camera_params.yaml."""
    focus_value:       float
    sensor_width:      int
    sensor_height:     int
    # Intrinsic matrix at sensor resolution
    fx: float
    fy: float
    cx: float
    cy: float
    # Distortion coefficients [k1, k2, p1, p2, k3]
    dist_coeffs: np.ndarray = field(repr=False)

    @property
    def sensor_matrix(self) -> np.ndarray:
        """3×3 camera matrix K at full sensor resolution."""
        return np.array([
            [self.fx,    0.0, self.cx],
            [0.0,    self.fy, self.cy],
            [0.0,        0.0,    1.0 ],
        ], dtype=np.float64)


# ─── YAML loader ──────────────────────────────────────────────────────────────

def load_params(yaml_path: str | Path) -> CalibrationParams:
    """
    Parse camera_params.yaml and return a CalibrationParams instance.

    Raises:
        FileNotFoundError: if the YAML file does not exist.
        KeyError / ValueError: if required fields are missing or malformed.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"camera_params.yaml not found: {path}")

    with path.open() as fh:
        data = yaml.safe_load(fh)

    meta  = data["metadata"]
    calib = data["calibration"]
    cm    = calib["camera_matrix"]

    dist_raw = calib["dist_coeffs"]
    if len(dist_raw) not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"dist_coeffs must have 4, 5, 8, 12, or 14 elements; got {len(dist_raw)}"
        )

    return CalibrationParams(
        focus_value  = float(meta["focus_value"]),
        sensor_width = int(meta["sensor_resolution"]["width"]),
        sensor_height= int(meta["sensor_resolution"]["height"]),
        fx           = float(cm["fx"]),
        fy           = float(cm["fy"]),
        cx           = float(cm["cx"]),
        cy           = float(cm["cy"]),
        dist_coeffs  = np.asarray(dist_raw, dtype=np.float64),
    )


# ─── Focus locking ────────────────────────────────────────────────────────────

def lock_focus_picamera2(picam2: "Picamera2", focus_value: float) -> None:
    """
    Disable continuous autofocus and hard-set the lens to *focus_value*
    using the Picamera2 / libcamera controls API.

    Pi Cam V3 (IMX708) lens range: ~0.0 (infinity) to ~10.0 (macro).

    Args:
        picam2:      A started Picamera2 instance.
        focus_value: Target lens position from camera_params.yaml.
    """
    picam2.set_controls({
        "AfMode":       _lc_controls.AfModeEnum.Manual,
        "LensPosition": focus_value,
    })
    logger.success("Focus locked via Picamera2 — LensPosition={}", focus_value)


def lock_focus_v4l2(device: str = "/dev/video0", focus_value: float = 0.0) -> None:
    """
    Fallback: lock focus via v4l2-ctl when Picamera2 is not available or
    the camera is accessed through a V4L2 node.

    v4l2 focus_absolute range for IMX708: 0–1023 (integer).
    Pass an integer focus_value when using this path.
    """
    focus_int = int(round(focus_value))
    try:
        subprocess.run(
            ["v4l2-ctl", f"--device={device}",
             "--set-ctrl=focus_automatic_continuous=0",
             f"--set-ctrl=focus_absolute={focus_int}"],
            check=True, capture_output=True,
        )
        logger.success("Focus locked via v4l2-ctl — focus_absolute={}", focus_int)
    except FileNotFoundError:
        logger.warning("v4l2-ctl not found; skipping focus lock (install v4l-utils)")
    except subprocess.CalledProcessError as exc:
        logger.warning("v4l2-ctl failed: {}; skipping focus lock", exc.stderr.decode())


# ─── Matrix scaling ───────────────────────────────────────────────────────────

def scale_camera_matrix(
    params: CalibrationParams,
    capture_width:  int,
    capture_height: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Scale the calibration matrix from sensor resolution to the chosen capture
    resolution.  Both axes are scaled independently so non-square cropping is
    handled correctly.

    Args:
        params:         Parsed CalibrationParams (intrinsics at sensor res).
        capture_width:  Width of frames produced by the camera at runtime.
        capture_height: Height of frames produced by the camera at runtime.

    Returns:
        K_scaled   — 3×3 numpy array, intrinsics at capture resolution.
        scale_x    — horizontal scale factor (capture_width  / sensor_width).
        scale_y    — vertical   scale factor (capture_height / sensor_height).
    """
    scale_x = capture_width  / params.sensor_width
    scale_y = capture_height / params.sensor_height

    if abs(scale_x - scale_y) > 1e-4:
        logger.warning(
            "Aspect ratios differ: sensor={}×{} capture={}×{} "
            "(scale_x={:.4f}, scale_y={:.4f}) — fx/cx and fy/cy scaled independently",
            params.sensor_width, params.sensor_height,
            capture_width, capture_height,
            scale_x, scale_y,
        )
    else:
        logger.info(
            "Matrix scaled ×{:.4f} ({}×{} → {}×{})",
            scale_x,
            params.sensor_width, params.sensor_height,
            capture_width, capture_height,
        )

    K_scaled = np.array([
        [params.fx * scale_x,           0.0, params.cx * scale_x],
        [          0.0,       params.fy * scale_y, params.cy * scale_y],
        [          0.0,                 0.0,                      1.0],
    ], dtype=np.float64)

    return K_scaled, scale_x, scale_y


# ─── Undistortion pipeline ────────────────────────────────────────────────────

class CameraUndistorter:
    """
    Reusable undistortion engine.

    Pre-computes remap look-up tables once at construction time so that
    each call to undistort_frame() is a single cv2.remap() — fast enough
    for real-time use.

    Args:
        yaml_path:    Path to camera_params.yaml.
        capture_size: (width, height) of frames captured at runtime.
                      If None, sensor resolution from the YAML is used.
        alpha:        getOptimalNewCameraMatrix alpha.
                      0.0 = all pixels valid (no black borders, FOV reduced).
                      1.0 = full sensor FOV retained (black borders present).
    """

    def __init__(
        self,
        yaml_path:    str | Path = "configs/camera_params.yaml",
        capture_size: Optional[Tuple[int, int]] = None,
        alpha:        float = 0.0,
    ) -> None:
        self.params = load_params(yaml_path)

        if capture_size is None:
            capture_size = (self.params.sensor_width, self.params.sensor_height)

        self.capture_width, self.capture_height = capture_size

        # Scale matrix to capture resolution
        self.K_scaled, self._sx, self._sy = scale_camera_matrix(
            self.params, self.capture_width, self.capture_height
        )
        self.dist = self.params.dist_coeffs

        # Optimal new matrix — alpha=0 crops all black edges
        self.K_new, self._roi = cv2.getOptimalNewCameraMatrix(
            self.K_scaled,
            self.dist,
            (self.capture_width, self.capture_height),
            alpha=alpha,
            newImgSize=(self.capture_width, self.capture_height),
        )

        # Pre-compute remap tables (much faster than cv2.undistort per frame)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K_scaled,
            self.dist,
            None,          # no rectification rotation
            self.K_new,
            (self.capture_width, self.capture_height),
            cv2.CV_16SC2,  # compact fixed-point maps — fastest remap type
        )

        rx, ry, rw, rh = self._roi
        logger.info(
            "CameraUndistorter ready — capture={}×{}, valid ROI={}×{} at ({},{})",
            self.capture_width, self.capture_height, rw, rh, rx, ry,
        )

    # ------------------------------------------------------------------ public

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply lens undistortion to *frame* and return the corrected image.

        The output is cropped to the valid ROI (all black borders removed when
        alpha=0 was used at construction).  Image dimensions may be smaller
        than the input.

        Args:
            frame: BGR uint8 image of shape (capture_height, capture_width, 3).

        Returns:
            Undistorted BGR image, cropped to the valid pixel region.
        """
        undistorted = cv2.remap(
            frame, self._map1, self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        x, y, w, h = self._roi
        return undistorted[y : y + h, x : x + w]

    @property
    def valid_roi(self) -> Tuple[int, int, int, int]:
        """(x, y, w, h) of the undistorted valid pixel region."""
        return self._roi

    @property
    def output_size(self) -> Tuple[int, int]:
        """(width, height) of the cropped undistorted output."""
        x, y, w, h = self._roi
        return w, h

    @property
    def new_camera_matrix(self) -> np.ndarray:
        """Optimal 3×3 camera matrix for the undistorted output."""
        return self.K_new


# ─── Picamera2 capture helpers ────────────────────────────────────────────────

def _make_picamera2(capture_width: int, capture_height: int) -> "Picamera2":
    """Configure and start a Picamera2 instance at the requested resolution."""
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (capture_width, capture_height), "format": "BGR888"},
        buffer_count=2,
    )
    picam2.configure(config)
    picam2.start()
    return picam2


def run_collect(
    undistorter: CameraUndistorter,
    output_dir:  str | Path,
    n_frames:    int = 50,
) -> None:
    """
    Capture *n_frames* undistorted images and write them to *output_dir*.
    Press SPACE to capture, Q to quit early.
    """
    if not _HAS_PICAMERA2:
        logger.error("Picamera2 not available — cannot run on this machine")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    picam2 = _make_picamera2(undistorter.capture_width, undistorter.capture_height)
    lock_focus_picamera2(picam2, undistorter.params.focus_value)

    saved = 0
    logger.info("Dataset collection — press SPACE to capture, Q to quit")
    try:
        while saved < n_frames:
            raw   = picam2.capture_array("main")
            clean = undistorter.undistort_frame(raw)

            cv2.imshow("Undistorted — SPACE=save  Q=quit", clean)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                fpath = out / f"frame_{saved:04d}.jpg"
                cv2.imwrite(str(fpath), clean, [cv2.IMWRITE_JPEG_QUALITY, 97])
                logger.info("Saved {}", fpath)
                saved += 1
            elif key == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
    logger.success("Collected {} frames → {}", saved, out)


def run_preview(undistorter: CameraUndistorter) -> None:
    """
    Live preview of the undistorted feed.  Press Q to quit.
    """
    if not _HAS_PICAMERA2:
        logger.error("Picamera2 not available — cannot run on this machine")
        return

    picam2 = _make_picamera2(undistorter.capture_width, undistorter.capture_height)
    lock_focus_picamera2(picam2, undistorter.params.focus_value)

    logger.info("Live preview — press Q to quit")
    try:
        while True:
            raw   = picam2.capture_array("main")
            clean = undistorter.undistort_frame(raw)
            cv2.imshow("Undistorted preview — Q=quit", clean)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Pi Camera V3 undistortion pipeline")
    p.add_argument("--yaml",   default="configs/camera_params.yaml",
                   help="Path to camera_params.yaml")
    p.add_argument("--width",  type=int, default=1920, help="Capture width")
    p.add_argument("--height", type=int, default=1080, help="Capture height")
    p.add_argument("--alpha",  type=float, default=0.0,
                   help="getOptimalNewCameraMatrix alpha (0=no borders, 1=full FOV)")
    p.add_argument("--mode",   choices=["preview", "collect"], default="preview")
    p.add_argument("--output", default="data/collected",
                   help="Output directory for 'collect' mode")
    p.add_argument("--frames", type=int, default=50,
                   help="Number of frames to collect in 'collect' mode")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    undistorter = CameraUndistorter(
        yaml_path    = args.yaml,
        capture_size = (args.width, args.height),
        alpha        = args.alpha,
    )

    if args.mode == "preview":
        run_preview(undistorter)
    else:
        run_collect(undistorter, args.output, args.frames)
