#!/usr/bin/env python3
"""
run_smirk_blank_on_no_face.py

Run SMIRK on a video but if Mediapipe doesn't find a face for a frame, do NOT reuse previous crop —
instead leave the cropped/renderer/generator output blank for that frame (black image). Continue processing.

Usage example:
python run_smirk_blank_on_no_face.py \
  --input_path audio_detection/dataset/video/1.mp4 \
  --out_path audio_detection/dataset/result_smirk \
  --checkpoint pretrained_models/SMIRK_em1.pt \
  --crop --render_orig --landmark_dir_name landmarks_01
"""
import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F


def crop_face_by_landmarks(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2]
    ])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    return tform


def center_blank_image(image_size):
    """Return a black RGB uint8 image of given square size."""
    return np.zeros((image_size, image_size, 3), dtype=np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')
    parser.add_argument('--landmark_dir_name', type=str, default='landmarks_00', help='Folder name under out_path where landmarks will be saved')

    args = parser.parse_args()

    input_image_size = 224

    # Device fallback
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    print(f"[INFO] Loading SMIRK encoder on device: {args.device}")
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    if args.use_smirk_generator:
        from src.smirk_generator import SmirkGenerator
        smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(args.device)

        checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k}
        smirk_generator.load_state_dict(checkpoint_generator)
        smirk_generator.eval()

        # load triangle probabilities for sampling points on the image
        face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)

    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        print(f'Error opening video file: {args.input_path}')
        exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.render_orig:
        out_width = video_width
        out_height = video_height
    else:
        out_width = input_image_size
        out_height = input_image_size

    if args.use_smirk_generator:
        out_width *= 3
    else:
        out_width *= 2

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    landmarks_folder = os.path.join(args.out_path, args.landmark_dir_name)
    os.makedirs(landmarks_folder, exist_ok=True)
    print(f"[INFO] Landmarks will be saved into: {landmarks_folder}")

    input_basename = os.path.basename(args.input_path)
    input_name_noext = os.path.splitext(input_basename)[0]
    out_video_path = os.path.join(args.out_path, f"{input_name_noext}.mp4")

    cap_out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (out_width, out_height))

    frame_counter = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break

        frame_counter += 1
        kpt_mediapipe = None
        try:
            kpt_mediapipe = run_mediapipe(image)
        except Exception as e:
            print(f"[WARN] run_mediapipe exception on frame {frame_counter}: {e}")
            kpt_mediapipe = None

        no_face = False  # flag for whether we have valid face landmarks this frame
        tform = None
        cropped_kpt_mediapipe = None

        # --- cropping decision ---
        if args.crop:
            if kpt_mediapipe is None:
                # NO landmarks detected -> produce blank cropped image and skip landmarks save
                no_face = True
                print(f"[WARN] Frame {frame_counter}: Mediapipe did not find landmarks. Leaving cropped/render blank for this frame.")
                cropped_image = center_blank_image(input_image_size)
                cropped_kpt_mediapipe = None
            else:
                # normal path: got landmarks
                kpt2 = kpt_mediapipe[..., :2]
                tform = crop_face_by_landmarks(image, kpt2, scale=1.4, image_size=input_image_size)
                cropped_image = warp(image, tform.inverse, output_shape=(input_image_size, input_image_size), preserve_range=True).astype(np.uint8)
                homog = np.hstack([kpt2, np.ones([kpt2.shape[0], 1])])
                cropped_kpt_mediapipe = (tform.params @ homog.T).T[:, :2]
        else:
            # not cropping: feed original frame (resized)
            cropped_image = image.copy()
            cropped_kpt_mediapipe = kpt_mediapipe

        # convert and normalize cropped image (or black if no face)
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image_rgb = cv2.resize(cropped_image_rgb, (input_image_size, input_image_size))
        cropped_image_t = torch.tensor(cropped_image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cropped_image_t = cropped_image_t.to(args.device)

        outputs = smirk_encoder(cropped_image_t)
        flame_output = flame.forward(outputs)

        # Save landmarks only if face detected (i.e., no_face == False)
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if not no_face:
            landmarks_2d = flame_output['landmarks_mp'].detach().cpu().numpy()[0]
            vertices_3d = flame_output['vertices'].detach().cpu().numpy()[0]
            np.save(os.path.join(landmarks_folder, f"frame_{frame_idx:04d}_2d.npy"), landmarks_2d)
            np.save(os.path.join(landmarks_folder, f"frame_{frame_idx:04d}_3d.npy"), vertices_3d)
            print(f"[INFO] Saved landmarks for frame {frame_idx} -> {landmarks_folder}")
        else:
            # Skip saving landmarks when no face
            pass

        renderer_output = renderer.forward(
            flame_output['vertices'], outputs['cam'],
            landmarks_fan=flame_output['landmarks_fan'],
            landmarks_mp=flame_output['landmarks_mp']
        )
        rendered_img = renderer_output['rendered_img']

        # If no face was detected, set rendered_img to zeros so the final grid shows blank
        if no_face:
            # create a zero tensor with same shape as rendered_img
            rendered_img = torch.zeros_like(rendered_img)

        # Build grid (original + rendered or cropped + rendered)
        if args.render_orig:
            if args.crop:
                # if we have tform (face detected), try warping rendered_img back; otherwise use zero image
                if not no_face and tform is not None:
                    try:
                        rendered_img_numpy = (rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                        rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)
                        rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    except Exception as e:
                        print(f"[WARN] Frame {frame_counter}: warping rendered image failed: {e}. Using black fallback.")
                        rendered_img_orig = torch.zeros((1, 3, video_height, video_width), dtype=torch.float32)
                else:
                    # blank rendered area
                    rendered_img_orig = torch.zeros((1, 3, video_height, video_width), dtype=torch.float32)
            else:
                rendered_img_orig = F.interpolate(rendered_img, (video_height, video_width), mode='bilinear').cpu()

            full_image = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            grid = torch.cat([full_image, rendered_img_orig], dim=3)
        else:
            # not rendering to original: concat cropped_image_t and rendered_img (both will be black if no_face)
            grid = torch.cat([cropped_image_t, rendered_img], dim=3)

        # SMIRK generator step: only run if enabled AND we have face keypoints (otherwise skip)
        if args.use_smirk_generator:
            if cropped_kpt_mediapipe is None:
                print(f"[WARN] Frame {frame_counter}: skipping smirk_generator (no mediapipe landmarks available).")
            else:
                mask_ratio_mul = 5
                mask_ratio = 0.01
                mask_dilation_radius = 10

                hull_mask = create_mask(cropped_kpt_mediapipe, (input_image_size, input_image_size))
                rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
                tmask_ratio = mask_ratio * mask_ratio_mul

                npoints, _ = masking_utils.mesh_based_mask_uniform_faces(
                    renderer_output['transformed_vertices'],
                    flame_faces=flame.faces_tensor,
                    face_probabilities=face_probabilities,
                    mask_ratio=tmask_ratio
                )

                pmask = torch.zeros_like(rendered_mask)
                rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
                rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
                rbound = (npoints.size(1) * (1 / mask_ratio_mul) * (rscale ** rsing)).long()

                for bi in range(npoints.size(0)):
                    pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1

                hull_mask = torch.from_numpy(hull_mask).type(dtype=torch.float32).unsqueeze(0).to(args.device)

                extra_points = cropped_image_t * pmask
                masked_img = masking_utils.masking(cropped_image_t, hull_mask, extra_points, mask_dilation_radius, rendered_mask=rendered_mask)

                smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)
                reconstructed_img = smirk_generator(smirk_generator_input)

                if args.render_orig:
                    if not no_face and tform is not None:
                        try:
                            reconstructed_img_numpy = (reconstructed_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                            reconstructed_img_orig = warp(reconstructed_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)
                            reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        except Exception as e:
                            print(f"[WARN] Frame {frame_counter}: warping reconstructed image failed: {e}. Using black fallback.")
                            reconstructed_img_orig = torch.zeros((1, 3, video_height, video_width), dtype=torch.float32)
                    else:
                        reconstructed_img_orig = torch.zeros((1, 3, video_height, video_width), dtype=torch.float32)

                    grid = torch.cat([grid, reconstructed_img_orig], dim=3)
                else:
                    grid = torch.cat([grid, reconstructed_img], dim=3)

        # convert grid to numpy and write frame
        grid_numpy = grid.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
        grid_numpy = grid_numpy.astype(np.uint8)
        grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)
        cap_out.write(grid_numpy)

    cap.release()
    cap_out.release()

    print(f"[DONE] Output video saved to: {out_video_path}")
    print(f"[DONE] Landmarks saved to: {landmarks_folder}")
