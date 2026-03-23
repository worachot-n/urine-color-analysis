"""
Two-class YOLOv8s training guide + OpenVINO export for Raspberry Pi 4B.

=============================================================================
CLASSES
=============================================================================
    Class 0: ref_bottle    — 15 reference bottles in the top row (known color standards)
    Class 1: sample_bottle — all test bottles placed in the 16x14 main grid

=============================================================================
DATASET STRUCTURE
=============================================================================
    dataset/
    ├── images/
    │   ├── train/          (150-300 images recommended)
    │   └── val/            (30-50 images recommended)
    ├── labels/
    │   ├── train/          (YOLO format .txt — one file per image)
    │   └── val/
    └── data.yaml

=============================================================================
data.yaml  (save as dataset/data.yaml)
=============================================================================
    path: /absolute/path/to/dataset
    train: images/train
    val:   images/val
    nc: 2
    names:
      0: ref_bottle
      1: sample_bottle

=============================================================================
ANNOTATION FORMAT  (YOLO .txt — one line per object)
=============================================================================
    <class_id> <x_center> <y_center> <width> <height>   (all 0-1 normalized)

    Examples:
      0 0.12 0.04 0.03 0.06   # ref_bottle in top row
      1 0.45 0.35 0.03 0.06   # sample_bottle in main grid

    Background images (empty grid): create an empty .txt file (zero bytes).
    YOLO treats images with no annotations as pure negative samples.

=============================================================================
DATASET DIVERSITY  (150-300 images minimum)
=============================================================================
    Distribution recommendation:
      70%  — bottles placed correctly (all levels L0-L4, various fill heights)
      20%  — empty grid (background suppression — unlabeled images)
      10%  — challenging cases:
               • Specular glare on bottle caps under direct LED light
               • Partially shadowed bottles near grid edges
               • Bottles at slight angles

    Always capture:
      • With LEDs on (same as production scanning environment)
      • From the exact camera mount position and angle
      • At various times of day (lighting changes)
      • Varying liquid levels L0 (clear) → L4 (dark amber)

=============================================================================
TRAINING COMMAND  (run on PC with CUDA GPU)
=============================================================================
    pip install ultralytics

    yolo train \\
        model=yolov8s.pt \\
        data=dataset/data.yaml \\
        epochs=100 \\
        imgsz=640 \\
        batch=16 \\
        mosaic=1.0 \\
        blur_p=0.3 \\
        hsv_v=0.4 \\
        hsv_s=0.5 \\
        hsv_h=0.015 \\
        flipud=0.0 \\
        fliplr=0.5 \\
        project=runs/urine_detector \\
        name=v1

    Best weights saved to: runs/urine_detector/v1/weights/best.pt

=============================================================================
EXPORT  (run this script after training)
=============================================================================
    python scripts/export_model.py

    Copy output directory to: models/bottle_yolo8s_openvino/ on Raspberry Pi

=============================================================================
INSTALLATION ON PI
=============================================================================
    pip install ultralytics openvino

=============================================================================
VERIFICATION
=============================================================================
    Run inference on one test image and confirm:
      - ref_bottle (cls=0) detected only in the top row
      - sample_bottle (cls=1) detected only in the main grid
      - Empty grid image → zero detections

    If ghosts remain on empty grid: add more background images to training set.
    If bottles missed: lower conf_threshold in config.toml (0.75 → 0.65).
"""

from ultralytics import YOLO

model = YOLO("runs/urine_detector/v1/weights/best.pt")

model.export(
    format="openvino",
    imgsz=640,
    half=False,       # Pi 4B CPU: FP32 — FP16 requires hardware FP16 support
    dynamic=False,
    simplify=True,
)

print("\nModel exported.")
print("Copy the output *_openvino_model/ directory to:")
print("  models/bottle_yolo8s_openvino/  on the Raspberry Pi")
