#!/usr/bin/env python3
"""
run.py

Python replacement for run_smirk_pipeline.sh.
- Runs face_crop_yolo.py to produce a cropped video.
- Runs demo_video.py on the cropped video and passes a customizable landmark folder name.

Usage:
  python3 run.py                             # uses defaults
  python3 run.py --input path/to/video.mp4 --output path/to/crop.mp4 --model yolov8n-face.pt \
                 --checkpoint pretrained_models/SMIRK_em1.pt --landmark_dir_name landmarks_custom \
                 --scale 0.25 --smoothing 0.6 --crop --render_orig

Notes:
- Requires face_crop_yolo.py and demo_video.py to be in the same folder (or give full paths).
- Uses the same arguments you'd provide in the shell script. See --help.
"""
import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_exists(path, name):
    if not Path(path).exists():
        print(f"[ERROR] {name} not found: {path}", file=sys.stderr)
        sys.exit(2)

def main():
    p = argparse.ArgumentParser(description="Run face crop then SMIRK demo (Python wrapper).")
    p.add_argument("--input", "-i", default="audio_detection/dataset/video/3.mp4",
                   help="Input video path (default audio_detection/dataset/video/3.mp4)")
    p.add_argument("--output", "-o", default="audio_detection/dataset/cropped_video/cropped_3.mp4",
                   help="Output (cropped) video path")
    p.add_argument("--model", "-m", default="yolov11m-face.pt", help="YOLO model path (default yolov11m-face.pt)")
    p.add_argument("--scale", type=float, default=0.25, help="Crop scale to pass to face_crop_yolo.py")
    p.add_argument("--smoothing", type=float, default=0.6, help="Smoothing alpha for face_crop_yolo.py")
    p.add_argument("--checkpoint", default="pretrained_models/SMIRK_em1.pt", help="SMIRK checkpoint (for demo_video.py)")
    p.add_argument("--out_path", default="audio_detection/dataset/result_smirk", help="demo_video.py out_path")
    p.add_argument("--landmark_dir_name", default="landmarks_00", help="Folder name under out_path to save landmarks")
    p.add_argument("--crop", action="store_true", help="Forward --crop to demo_video.py")
    p.add_argument("--render_orig", action="store_true", help="Forward --render_orig to demo_video.py")
    p.add_argument("--skip_face_crop", action="store_true", help="Skip running face_crop_yolo.py and use provided --output as input for demo_video.py")
    p.add_argument("--face_crop_script", default="face_crop_yolo.py", help="Path to face_crop_yolo.py")
    p.add_argument("--demo_script", default="demo_video.py", help="Path to demo_video.py")
    p.add_argument("--python_bin", default=sys.executable, help="Python interpreter to use (default: this python)")

    args = p.parse_args()

    # Resolve paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)
    checkpoint_path = Path(args.checkpoint)
    out_path = Path(args.out_path)
    face_crop_script = Path(args.face_crop_script)
    demo_script = Path(args.demo_script)
    python_bin = args.python_bin

    # sanity checks
    check_exists(input_path, "Input video")
    check_exists(face_crop_script, "face_crop script")
    check_exists(demo_script, "demo script")
    check_exists(checkpoint_path, "SMIRK checkpoint")
    # model may be optional if face_crop_yolo supports auto-download; but we still warn if missing
    if not model_path.exists():
        print(f"[WARN] YOLO model not found at {model_path}. Make sure this is intentional.", file=sys.stderr)

    # ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input        : {input_path}")
    print(f"[INFO] Cropped out  : {output_path}")
    print(f"[INFO] YOLO model   : {model_path}")
    print(f"[INFO] SMIRK ckpt   : {checkpoint_path}")
    print(f"[INFO] demo out path: {out_path}")
    print(f"[INFO] landmark dir : {args.landmark_dir_name}")
    print()

    try:
        if not args.skip_face_crop:
            # build face_crop command
            face_crop_cmd = [
                python_bin, str(face_crop_script),
                "--input", str(input_path),
                "--output", str(output_path),
                "--model", str(model_path),
                "--scale", str(args.scale),
                "--smoothing", str(args.smoothing)
            ]
            print("[INFO] Running face cropping:")
            print(" ".join(face_crop_cmd))
            subprocess.run(face_crop_cmd, check=True)
            print("[INFO] Face cropping finished.\n")
        else:
            print("[INFO] Skipping face crop step (--skip_face_crop set). Using provided output as demo input.\n")

        # build demo command
        demo_cmd = [
            python_bin, str(demo_script),
            "--input_path", str(output_path),
            "--out_path", str(out_path),
            "--checkpoint", str(checkpoint_path),
            "--landmark_dir_name", args.landmark_dir_name
        ]
        if args.crop:
            demo_cmd.append("--crop")
        if args.render_orig:
            demo_cmd.append("--render_orig")

        print("[INFO] Running SMIRK demo:")
        print(" ".join(demo_cmd))
        subprocess.run(demo_cmd, check=True)
        print("[INFO] SMIRK demo finished.\n")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
        sys.exit(1)

    print("[DONE] Pipeline finished. Results in:", out_path)
    print("[DONE] Landmarks saved to:", out_path / args.landmark_dir_name)


if __name__ == "__main__":
    main()
