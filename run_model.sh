#!/usr/bin/env bash
set -euo pipefail
# run_smirk_pipeline.sh
# Usage:
#   ./run_smirk_pipeline.sh [INPUT_VIDEO] [CROPPED_VIDEO_OUTPUT] [YOLO_MODEL_PATH]
# Example:
#   ./run_smirk_pipeline.sh \
#     audio_detection/dataset/video/2.mp4 \
#     audio_detection/dataset/cropped_video/cropped_2.mp4 \
#     pretrained_models/yolov8n-face.pt

# --- defaults (will be used if positional args are not provided) ---
INPUT="${1:-audio_detection/dataset/video/3.mp4}"
OUTPUT="${2:-audio_detection/dataset/cropped_video/cropped_3.mp4}"
MODEL_PATH="${3:-yolov11m-face.pt}"   # keep your default; change if needed

# create parent directories if missing
mkdir -p "$(dirname "$OUTPUT")"
mkdir -p audio_detection/dataset/result_smirk

echo "[INFO] Input:  $INPUT"
echo "[INFO] Output: $OUTPUT"
echo "[INFO] YOLO model: $MODEL_PATH"

# Run face cropping (use python3)
python3 face_crop_yolo.py --input "$INPUT" --output "$OUTPUT" --model "$MODEL_PATH" --scale 0.25 --smoothing 0.6

# Run SMIRK demo on the cropped video (pass the cropped video as input)
python3 demo_video.py \
  --input_path "$OUTPUT" \
  --out_path audio_detection/dataset/result_smirk \
  --checkpoint pretrained_models/SMIRK_em1.pt \
  --crop \
  --render_orig

echo "[DONE] Pipeline finished. Results in audio_detection/dataset/result_smirk"
