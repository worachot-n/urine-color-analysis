"""
Export YOLOv11s to OpenVINO format for Raspberry Pi 4B deployment.

Usage (run on PC with GPU, then copy model directory to Pi):
    python scripts/export_model.py

Output:
    yolo11s_openvino_model/   (created in the current directory)

Deployment:
    Copy the output directory to: models/bottle_yolo11s_openvino/ on the Pi
    The directory must contain: yolo11s.xml, yolo11s.bin, metadata.yaml

Install on Pi before running main.py:
    pip install ultralytics openvino

-------------------------------------------------------------------------------
Training advice — Background Samples (Anti-Ghosting)
-------------------------------------------------------------------------------

To suppress reflections, grid intersections, and printed-line false positives:

1. Collect 50-100 images of the EMPTY grid under normal operating conditions
   (LEDs on, same mount, same camera angle as production scans).

2. Save these images with EMPTY annotation files (zero-byte .txt files or
   YOLO label files with no lines). YOLOv11 treats images with no annotations
   as pure negative/background samples automatically.

3. Recommended dataset composition:
     70%  — bottles in correct slots (varied fills, angles, lighting)
     20%  — empty grid images (background suppression)
     10%  — challenging cases (partial glare, edge slots, reflections)

4. Augmentations to enable in data.yaml:
     hsv_h: 0.015    # slight hue shift
     hsv_s: 0.5      # saturation variation (LED intensity changes)
     hsv_v: 0.4      # brightness variation (simulate different lighting)
     flipud: 0.0     # DON'T flip vertically — camera is fixed
     fliplr: 0.5     # horizontal flip is safe (symmetric grid)
     mosaic: 0.5

5. After training, verify on an empty grid photo:
   inference should produce zero detections.
"""

from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.export(
    format="openvino",
    imgsz=640,
    half=False,       # Pi 4B CPU: use FP32 — FP16 requires hardware support
    dynamic=False,
    simplify=True,
)

print("\nModel exported to: yolo11s_openvino_model/")
print("Copy the folder to: models/bottle_yolo11s_openvino/ on the Raspberry Pi")
