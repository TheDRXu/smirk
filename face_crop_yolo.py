import argparse
import os
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Detect faces with YOLO and produce cropped/resized output.")
    p.add_argument("--input", "-i", required=True, help="Input video file path")
    p.add_argument("--output", "-o", required=False, help="Output video file path (mp4). If omitted and --frames_out omitted, prints to stdout.")
    p.add_argument("--frames_out", "-f", required=False, help="Directory to save cropped frames (jpg). If provided, frames will be saved. Can be used with --output as well.")
    p.add_argument("--model", "-m", default="yolov11m-face.pt", help="YOLO model path (default: yolov8n-face.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    p.add_argument("--confidence", "-c", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--scale", "-s", type=float, default=0.25, help="Crop scale relative to original frame width (0.25 = 25%% of original width)")
    p.add_argument("--pad_rel", type=float, default=0.10, help="Relative padding applied to detection boxes when they are used for crop size")
    p.add_argument("--smoothing", type=float, default=0.6, help="EMA smoothing alpha in [0,1]. 0=no smoothing, closer to 1 = more smoothing")
    p.add_argument("--pick_largest", action="store_true", help="If multiple faces, pick the largest (default True behavior). Use --no-pick to pick first; this flag enforces pick largest.")
    p.add_argument("--no_gpu", action="store_true", help="Force CPU (Ultralytics may still pick GPU by default)")
    p.add_argument("--show_debug", action="store_true", help="Show debug windows (requires GUI; not for headless servers)")
    p.add_argument("--skip_no_face", action="store_true", help="If no face is detected in a frame, skip writing that frame (default = reuse last crop / center fallback).")
    return p.parse_args()

def clamp(v, a, b):
    return max(a, min(b, v))

def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input video not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    if args.frames_out:
        frames_out_dir = Path(args.frames_out)
        frames_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_out_dir = None

    if args.output:
        out_path = Path(args.output)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    device = None
    if args.no_gpu:
        device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[INFO] Forcing CPU mode (CUDA_VISIBLE_DEVICES cleared).")
    else:
        device = None  # ultralytics will auto-select GPU if available

    # Load YOLO model
    print(f"[INFO] Loading model: {args.model}")
    try:
        model = YOLO(args.model) if device is None else YOLO(args.model, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to load model '{args.model}': {e}", file=sys.stderr)
        sys.exit(3)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}", file=sys.stderr)
        sys.exit(4)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # target crop size (preserve frame aspect ratio)
    target_w = max(1, int(W * args.scale))
    # maintain same aspect ratio (target_h scaled from target_w)
    target_h = max(1, int(target_w * (H / max(1, W))))

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (target_w, target_h))
        if not writer.isOpened():
            print(f"[ERROR] Could not open video writer for {out_path}", file=sys.stderr)
            writer = None

    smoothed_cx = smoothed_cy = smoothed_w = smoothed_h = None
    frame_idx = 0
    last_written = 0
    start_time = time.time()

    print(f"[INFO] Input: {W}x{H} px, {FPS:.2f} FPS, frames={total_frames}")
    print(f"[INFO] Output crop size: {target_w}x{target_h} px (scale={args.scale})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # run YOLO on this frame
            # pass verbose=False to suppress extra prints
            results = model(frame, imgsz=args.imgsz, conf=args.confidence, verbose=False)
            r = results[0]

            boxes = []
            # r.boxes is a list of BoxObjects; convert to tuples (x1,y1,x2,y2,conf)
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int).flatten()[:4]
                conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") else 1.0
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                boxes.append((x1, y1, x2, y2, conf))

            if len(boxes) == 0:
                # No face detected
                if smoothed_cx is None:
                    # no previous crop: fallback center crop
                    cx = W / 2.0
                    cy = H / 2.0
                    crop_w = target_w
                    crop_h = target_h
                    reuse_previous = False
                else:
                    # reuse previous smoothed crop unless user explicitly wants to skip
                    if args.skip_no_face:
                        # skip writing for this frame
                        if frame_idx % 50 == 0:
                            print(f"[WARN] frame {frame_idx}: no face detected - skipping (skip_no_face True)")
                        continue
                    cx, cy, crop_w, crop_h = smoothed_cx, smoothed_cy, smoothed_w, smoothed_h
                    reuse_previous = True
            else:
                # choose the face to follow
                if args.pick_largest:
                    best = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                else:
                    best = boxes[0]
                x1, y1, x2, y2, conf = best
                det_w = x2 - x1
                det_h = y2 - y1
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # pick crop size
                if det_w >= target_w or det_h >= target_h:
                    crop_w = int(det_w * (1.0 + args.pad_rel))
                    crop_h = int(det_h * (1.0 + args.pad_rel))
                else:
                    crop_w, crop_h = target_w, target_h
                reuse_previous = False

            # smoothing (EMA)
            if smoothed_cx is None:
                smoothed_cx, smoothed_cy, smoothed_w, smoothed_h = cx, cy, int(crop_w), int(crop_h)
            else:
                a = clamp(args.smoothing, 0.0, 0.9999)
                smoothed_cx = a * smoothed_cx + (1 - a) * cx
                smoothed_cy = a * smoothed_cy + (1 - a) * cy
                smoothed_w = int(round(a * smoothed_w + (1 - a) * crop_w))
                smoothed_h = int(round(a * smoothed_h + (1 - a) * crop_h))

            # final crop coords (centered)
            cx_f = smoothed_cx
            cy_f = smoothed_cy
            crop_w_f = max(1, int(smoothed_w))
            crop_h_f = max(1, int(smoothed_h))

            x1c = int(round(cx_f - crop_w_f / 2.0))
            y1c = int(round(cy_f - crop_h_f / 2.0))
            x2c = x1c + crop_w_f
            y2c = y1c + crop_h_f

            # clamp to image bounds; if crop smaller than target because near border, we'll pad by adjusting the crop window
            if x1c < 0:
                x1c = 0
                x2c = clamp(crop_w_f, 1, W)
            if y1c < 0:
                y1c = 0
                y2c = clamp(crop_h_f, 1, H)
            if x2c > W:
                x2c = W
                x1c = max(0, W - crop_w_f)
            if y2c > H:
                y2c = H
                y1c = max(0, H - crop_h_f)

            # ensure valid region
            crop_w_final = max(1, x2c - x1c)
            crop_h_final = max(1, y2c - y1c)
            crop = frame[y1c:y2c, x1c:x2c]

            # If crop doesn't match desired aspect (rare), we still resize to target_w x target_h
            final = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # write frame to video writer if requested
            written = False
            if writer:
                writer.write(final)
                written = True

            # save individual frame if requested
            if frames_out_dir:
                out_file = frames_out_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_file), final)
                written = True

            last_written = frame_idx if written else last_written

            # optional debug display
            if args.show_debug:
                display = frame.copy()
                cv2.rectangle(display, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
                cv2.putText(display, f"Frame {frame_idx}/{total_frames}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow("orig_debug", display)
                cv2.imshow("crop", final)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # lightweight progress printing
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_time
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                print(f"[PROGRESS] frame {frame_idx}/{total_frames} | written_to_output={written} | proc_fps={fps_proc:.1f}")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show_debug:
            cv2.destroyAllWindows()

    print(f"[DONE] Processed {frame_idx} frames. Last written frame: {last_written}.")
    if out_path:
        print(f"[INFO] Output video: {out_path}")
    if frames_out_dir:
        print(f"[INFO] Saved frames: {frames_out_dir}")

if __name__ == "__main__":
    main()
