#!/usr/bin/env python3
"""
overlay_speaker_labels_with_audio.py

Overlay speaker role (Doctor / Patient) on a video according to a CSV of segments,
and attach WAV audio to the output.

CSV required columns: start,end,pred,prob_doctor

Example:
  python3 overlay_speaker_label.py \
    --video video.mp4 \
    --csv segments_predictions.csv \
    --wav audio.wav \
    --out labeled_with_audio.mp4

This script writes a temporary video without audio, then merges audio with ffmpeg.
If ffmpeg is not available it falls back to moviepy (slower).

Requirements:
  pip install pandas opencv-python numpy moviepy
  ffmpeg (recommended) — installable via system package manager or from https://ffmpeg.org/

"""
import argparse
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import cv2
import pandas as pd
import numpy as np
import math
import os

def load_segments(csv_path):
    df = pd.read_csv(csv_path)
    for c in ['start','end','pred','prob_doctor']:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    segs = []
    for _, r in df.iterrows():
        s = float(r['start']); e = float(r['end'])
        pred = str(r['pred'])
        prob = float(r['prob_doctor'])
        segs.append({'start': s, 'end': e, 'pred': pred, 'prob': prob})
    return segs

def active_segment_for_time(segs, t, frame_dt):
    best = None
    best_overlap = 0.0
    for seg in segs:
        s, e = seg['start'], seg['end']
        ov = max(0.0, min(e, t + frame_dt) - max(s, t))
        if ov > best_overlap:
            best_overlap = ov
            best = seg
        elif ov == best_overlap and best is not None:
            if seg['prob'] > best['prob']:
                best = seg
    return best

def draw_label(frame, text, prob, position=(20,40), box_color=(0,160,255), opacity=0.75):
    x,y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    txt = f"{text} {prob:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
    pad_x = 10
    pad_y = 6
    x2 = x + tw + pad_x*2
    y2 = y + th + pad_y*2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), box_color, -1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    text_x = x + pad_x
    text_y = y + pad_y + th
    cv2.putText(frame, txt, (text_x, text_y), font, font_scale, (255,255,255), thickness, lineType=cv2.LINE_AA)

def mux_audio_with_ffmpeg(video_in, wav_in, video_out):
    # re-encode video to libx264 and encode audio aac; -shortest to avoid mismatch
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_in),
        "-i", str(wav_in),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(video_out)
    ]
    print("[INFO] Running ffmpeg to mux audio (this will re-encode video):")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("[ERROR] ffmpeg failed. stderr:")
        print(proc.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError("ffmpeg failed to mux audio")
    return True

def mux_audio_with_moviepy(video_in, wav_in, video_out):
    from moviepy.editor import VideoFileClip, AudioFileClip
    print("[INFO] Using moviepy fallback to attach audio (slower).")
    vc = VideoFileClip(str(video_in))
    ac = AudioFileClip(str(wav_in))
    vc2 = vc.set_audio(ac)
    vc2.write_videofile(str(video_out), codec='libx264', audio_codec='aac', threads=4, verbose=False)
    vc.close(); ac.close()

def process(video_path, csv_path, wav_path, out_path, show=False, label_pos=(20,20)):
    segs = load_segments(csv_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_dt = 1.0 / fps

    # temporary output (no audio)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4v is broadly compatible; we'll re-encode with ffmpeg later
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open temp video writer: {tmp_path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_idx / fps
            seg = active_segment_for_time(segs, t, frame_dt)

            if seg is not None:
                label = seg['pred']
                prob = seg['prob']
                col = (30,160,30) if label.lower().startswith('d') else (0,120,200)
                draw_label(frame, label, prob, position=(int(label_pos[0]), int(label_pos[1])), box_color=col)

            writer.write(frame)

            if show:
                cv2.imshow("preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Early exit requested.")
                    break

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    # Now mux audio
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    try:
        if ffmpeg_path:
            mux_audio_with_ffmpeg(tmp_path, wav_path, out_path)
        else:
            # fallback to moviepy
            print("[WARN] ffmpeg not found — falling back to moviepy (slower). Install ffmpeg for best performance.")
            mux_audio_with_moviepy(tmp_path, wav_path, out_path)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

    print("[DONE] wrote labeled + audio video to:", out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Input video file (mp4)")
    p.add_argument("--csv", required=True, help="CSV with columns: start,end,pred,prob_doctor")
    p.add_argument("--wav", required=True, help="WAV audio file to attach to output")
    p.add_argument("--out", required=True, help="Output mp4 file path")
    p.add_argument("--show", action="store_true", help="Show preview while processing")
    p.add_argument("--label_x", type=int, default=20, help="Label X position (px)")
    p.add_argument("--label_y", type=int, default=20, help="Label Y position (px)")
    args = p.parse_args()

    process(args.video, args.csv, args.wav, args.out, show=args.show, label_pos=(args.label_x, args.label_y))

if __name__ == "__main__":
    main()
