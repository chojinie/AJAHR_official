import os
from pathlib import Path
import torch
import torch.nn.functional as F
import argparse
import os
import cv2
import numpy as np
import tqdm
from icecream import ic

from lib.models import load_ajahr
from lib.utils import recursive_to
from lib.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from lib.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
GREEN =(0.4039, 0.7294, 0.2392)

import inspect
import lib.utils.renderer as R

print("Renderer module file:", R.__file__)
print("Renderer.render_rgba_multiple source file:",
      inspect.getsourcefile(R.Renderer.render_rgba_multiple))


def main():
    parser = argparse.ArgumentParser(description='AJAHR demo code')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to pretrained model checkpoint')
    parser.add_argument('--model_config', type=str, default='model_config.yaml', help='Path to model config file')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out/', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')

    args = parser.parse_args()

    model, model_cfg = load_ajahr(checkpoint_path=args.checkpoint, \
                                     model_cfg=args.model_config, \
                                     is_train_state=False, is_demo=True)

    # Setup model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    kp_detector = ViTPoseModel(device)
    

    # Load detector
    from detectron2.engine.defaults import DefaultPredictor
    from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import lib
    cfg_path = Path(lib.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    img_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        img_paths.extend(list(Path(args.img_folder).rglob(ext)))

    # Iterate over all images in folder
    # for img_path in tqdm.tqdm(sorted(Path(args.img_folder).glob('*'))):
    for img_path in tqdm.tqdm(img_paths):
        ic(f'Processing {img_path}..')
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()


        # Detect keypoints for each person. # 키포인트 찾음.
        vitposes_out = kp_detector.predict_pose(
            img_cv2[:, :, ::-1],
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        vitposes_list = [vitpose["keypoints"] for vitpose in vitposes_out]
        
        try:    
            body_keypoints_2d = np.stack(vitposes_list)
        except ValueError:
            continue
        
        low_conf_mask = body_keypoints_2d[..., 2] < 0.55

        body_keypoints_2d[low_conf_mask, :2] = 0.0

        body_keypoints_2d[low_conf_mask, 2] = -1.0

        # Run on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, pred_bboxes, body_keypoints_2d)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            pred_cam = out['pred_cam']

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
           
            cam_view, depth = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)

            input_rgb = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0  # (H,W,3)

            cam_rgb = cam_view[:, :, :3].astype(np.float32)

            alpha = (depth > 0).astype(np.float32)
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            alpha = alpha[..., None]

            input_img_overlay = input_rgb * (1.0 - alpha) + cam_rgb * alpha
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.png'),
                        (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8))


if __name__ == '__main__':
    main()
