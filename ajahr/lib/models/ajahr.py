from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode

from ..utils.geometry import perspective_projection
from ..utils.misc import load_pretrained
from .backbones import create_backbone
from .heads import build_smpl_head, build_classifier
from .smpl_wrapper import SMPL


class AJAHR(pl.LightningModule):
    """
    DEMO / INFERENCE ONLY
    - classifier -> amputation mask -> smpl_head -> SMPL forward -> 2D/3D outputs
    - train/val/loss/optimizer/logging 전부 제거
    """

    def __init__(self, cfg: CfgNode, load_ckpt=True):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)

        # --- build modules ---
        self.backbone = create_backbone(cfg, load_weights=False)
        self.smpl_head = build_smpl_head(cfg)
        self.classifier = build_classifier()
                
        if load_ckpt:
            self.backbone, self.smpl_head, self.classifier = load_pretrained(
                cfg, self.backbone, self.smpl_head, self.classifier, is_train_state=False
            )

        self.backbone.eval()
        self.smpl_head.eval()
        self.classifier.eval()

        # --- SMPL ---
        smpl_cfg = {k.lower(): v for k, v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        # --- heads ---
        self.head_names = ['head1', 'head2', 'head3', 'head4']

        # --- head range (AJAHR classes) ---
        self.head_ranges = {
            'head1': (0, 1, 2),
            'head2': (3, 4, 5),
            'head3': (6, 7, 8),
            'head4': (9, 10, 11)
        }

    # -------------------------
    # utils (demo용 최소)
    # -------------------------
    @torch.no_grad()
    def get_amputation_prediction(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        logits_dict: {'head1':(B,4), ...}
        pred < 3 => 1 (amputation), pred == 3 => 0 (normal)
        return: (B,4)
        """
        preds = []
        for h in self.head_names:
            ph = torch.argmax(logits_dict[h], dim=1)
            preds.append((ph < 3).long())
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def map_pred_to_smpl_indices(self, logits_dict: Dict[str, torch.Tensor]) -> List[List[int]]:
        """
        demo에서는 label GT가 없어도 되므로,
        classifier 예측(argmax) 기반으로 zero-out 할 SMPL joint index를 만든다.
        """
        pred_h = {h: torch.argmax(logits_dict[h], dim=1) for h in self.head_names}

        mapping1 = {0: [20, 22], 1: [18, 20, 22], 2: [16, 18, 20, 22], 3: []}
        mapping2 = {0: [19, 21], 1: [17, 19, 21], 2: [15, 17, 19, 21], 3: []}
        mapping3 = {0: [6, 9],   1: [3, 6, 9],     2: [0, 3, 6, 9],     3: []}
        mapping4 = {0: [7, 10],  1: [4, 7, 10],    2: [1, 4, 7, 10],    3: []}

        B = pred_h['head1'].shape[0]
        out: List[List[int]] = []
        for i in range(B):
            out.append(
                mapping1[pred_h['head1'][i].item()]
                + mapping2[pred_h['head2'][i].item()]
                + mapping3[pred_h['head3'][i].item()]
                + mapping4[pred_h['head4'][i].item()]
            )
        return out

    def create_mask(self, B: int, J: int, smpl_index_list: List[List[int]], device) -> torch.Tensor:
        """
        return: (B,J) bool
        """
        mask = torch.zeros(B, J, dtype=torch.bool, device=device)
        for b in range(B):
            idxs = smpl_index_list[b]
            if len(idxs) > 0:
                mask[b, idxs] = True
        return mask

    # -------------------------
    # demo forward
    # -------------------------
    @torch.no_grad()
    def forward(self, batch: Dict) -> Dict:
        """
        required:
          batch['img']        : (B,3,H,W)
          batch['keypoints_2d']: (B,K,3)  (여기서는 5:17만 사용)

        returns:
          - classifier logits/preds
          - smpl outputs (vertices, kpts3d, kpts2d)
        """
        x = batch['img']
        B = x.shape[0]

        # 1) backbone
        feats = self.backbone(x)

        # 2) classifier
        kp2d = batch['keypoints_2d'][:, 5:17].float()
        logits_dict, cls_features = self.classifier(x, kp2d)  # (dict, feature)

        # 3) amputation pred (B,4)
        amp_pred = self.get_amputation_prediction(logits_dict)

        # 4) SMPL head (mask + feature conditioning)
        pred_smpl_params, pred_cam, _ = self.smpl_head(feats, mask=amp_pred, y=cls_features)

        # 5) camera translation
        device = pred_cam.device
        dtype = pred_cam.dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(B, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )

        # 6) reshape pose matrices
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(B, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(B, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(B, -1)

        # 7) zero-out predicted amputated joints (demo 기준: 예측 기반)
        smpl_index_list = self.map_pred_to_smpl_indices(logits_dict)

        Bp, J, a, b = pred_smpl_params['body_pose'].shape
        pose_flat = pred_smpl_params['body_pose'].reshape(Bp, J, -1)  # (B,J,9)

        mask = self.create_mask(Bp, J, smpl_index_list, device).unsqueeze(-1).expand_as(pose_flat)
        pose_flat[mask] = 0.0
        pred_smpl_params['body_pose'] = pose_flat.view(Bp, J, a, b)

        # 8) SMPL forward
        smpl_output = self.smpl(**{k: v for k, v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints.reshape(B, -1, 3)
        pred_vertices = smpl_output.vertices.reshape(B, -1, 3)

        # 9) projection to 2D
        pred_keypoints_2d = perspective_projection(
            smpl_output.joints,
            translation=pred_cam_t,
            focal_length=(focal_length / self.cfg.MODEL.IMAGE_SIZE),
        ).reshape(B, -1, 2)

        # 10) head별 pred 반환(보기 좋게)
        head_pred = {h: torch.argmax(logits_dict[h], dim=1) for h in self.head_names}

        return {
            "logits": logits_dict,          # head별 (B,4)
            "head_pred": head_pred,         # head별 (B,)
            "amp_pred": amp_pred,           # (B,4)
            "pred_cam": pred_cam,           # (B,3)
            "pred_cam_t": pred_cam_t,       # (B,3)
            "pred_smpl_params": {k: v for k, v in pred_smpl_params.items()},
            "pred_keypoints_3d": pred_keypoints_3d,
            "pred_vertices": pred_vertices,
            "pred_keypoints_2d": pred_keypoints_2d,
            "smpl_zero_indices": smpl_index_list,  # 배치별 zero-out 인덱스 기록(디버깅용)
        }
