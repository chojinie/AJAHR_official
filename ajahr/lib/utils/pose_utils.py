"""
Code adapted from: https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py
"""
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
from .rotation_utils import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_euler_angles
# keypoints 2D 44 = 25 Openpose + 14 lsp + 5?
JOINT44_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'Nose',
'Neck',
'RShoulder',
'RElbow',
'RWrist',
'LShoulder',
'LElbow',
'LWrist',
'MidHip',
'RHip',
'RKnee',
'RAnkle',
'LHip',
'LKnee',
'LAnkle',
'REye',
'LEye',
'REar',
'LEar',
'LBigToe',
'LSmallToe',
'LHeel',
'RBigToe',
'RSmallToe',
'RHeel',
# 14 LSP joints
'R_Ankle',
'R_Knee',
'R_Hip',
'L_Hip',
'L_Knee',
'L_Ankle',
'R_Wrist',
'R_Elbow',
'R_Shoulder',
'L_Shoulder',
'L_Elbow',
'L_Wrist',
'Neck_LSP',
'HeadTop_LSP',

'Pelvis_MPII',
'Thorax_MPII',
'Spine_H36M',
'Jaw_H36M',
'Head_H36M',
'Nose_other',
]

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    # Ensure that the input is of type float32
    S1 = S1.to(torch.float32)
    S2 = S2.to(torch.float32)

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re

def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    # Reconstruction_error
    r_error = reconstruction_error(pred_joints, gt_joints).cpu().numpy()
    return 1000 * mpjpe, 1000 * r_error

class Evaluator:
    def __init__(self,
                    dataset_length: int,
                    keypoint_list: List,
                    pelvis_ind: int,
                    metrics: List = ['mode_mpjpe', 'mode_re', 'mode_pve'],
                    J_regressor_24_SMPL = None,
                    dataset=''):
            """
            Class used for evaluating trained models on different 3D pose datasets.
            """
            self.dataset_length = dataset_length
            self.keypoint_list = keypoint_list
            self.pelvis_ind = pelvis_ind
            self.metrics = metrics
            self.J_regressor_24_SMPL = J_regressor_24_SMPL
            self.dataset = dataset
            for metric in self.metrics:
                setattr(self, metric, np.zeros((dataset_length,)))
            self.counter = 0

            self.imgnames = []

            # 분류(classification) metric 저장용: 배치별 accuracy, f1, 예측 및 정답값 저장
            self.per_head_accuracy = {}
            self.per_head_f1 = {}
            self.per_head_preds = {}   # head별 예측값 전체 저장 (각 배치별 numpy 배열 누적)
            self.per_head_labels = {}  # head별 정답값 전체 저장
            self.vertex = {
                'gt' : [],
                'pred' : []
            }
            self.per_head_precision = {}
            self.per_head_recall = {}

    def log(self):
        """
        Print current evaluation metrics, including pose and classification metrics.
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return

        print(f'{self.counter} / {self.dataset_length} samples processed')
        # 기존 pose 관련 metric 출력
        for metric in self.metrics:
            unit = 'mm' if metric in ['mode_mpjpe', 'mode_re', 'mode_pve'] else ''
            metric_val = getattr(self, metric)[:self.counter].mean()
            print(f'{metric}: {metric_val:.4f} {unit}')

        # classification 관련 metric 출력 (head별)
        if self.per_head_accuracy:
            print("Classification Metrics:")
            for head in self.per_head_accuracy.keys():
                avg_acc = np.mean(self.per_head_accuracy[head])
                avg_f1 = np.mean(self.per_head_f1[head])
                print(f'Head {head} -> Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}')
            
        if self.per_head_precision:
            print("Precision_Recall Metrics")
            for head in self.per_head_precision.keys():
                avg_precision = np.mean(self.per_head_precision[head])
                print(f'Head {head} -> Precision : {avg_precision:.4f}')
        
        if self.per_head_recall:
            print("Precision_Recall Metrics")
            for head in self.per_head_recall.keys():
                avg_recall = np.mean(self.per_head_recall[head])
                print(f'Head {head} -> Recall: {avg_recall:.4f}')
        
                    
        print('***')

    def get_metrics_dict(self) -> Dict:
        """
        Returns:
            Dict: Dictionary of evaluation metrics.
        """
        d1 = {metric: getattr(self, metric)[:self.counter].mean() for metric in self.metrics}
        if self.per_head_accuracy:
            for head in self.per_head_accuracy.keys():
                d1[f'{head}_accuracy'] = np.mean(self.per_head_accuracy[head])
                d1[f'{head}_f1'] = np.mean(self.per_head_f1[head])
        return d1


    def compute_confusion_matrices(self) -> Dict:
        """
        모든 배치에서 저장된 분류 결과로부터 각 head별 confusion matrix를 계산하여 반환.
        Returns:
            Dict: head 이름을 key로 하고 confusion matrix (np.array)를 value로 갖는 딕셔너리.
        """
        confusion_matrices = {}
        for head in self.per_head_preds:
            # 각 head별 리스트에 저장된 np.array를 concatenate하여 전체 결과로 만듦.
            all_preds = np.concatenate(self.per_head_preds[head])
            all_labels = np.concatenate(self.per_head_labels[head])
            cm = confusion_matrix(all_labels, all_preds)
            confusion_matrices[head] = cm
        return confusion_matrices

    def compute_precision_recall(self) -> Dict:
        """
        모든 head별로 precision과 recall (weighted average)을 계산하여 반환합니다.
        Returns:
            Dict: head 이름을 key로 하고 (precision, recall) 튜플을 value로 갖는 딕셔너리.
        """
        pr_dict = {}
        for head in self.per_head_preds:
            all_preds = np.concatenate(self.per_head_preds[head])
            all_labels = np.concatenate(self.per_head_labels[head])
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            pr_dict[head] = (precision, recall)
        return pr_dict


    def save_confusion_matrix_plots(self, output_dir="confusion_matrices", file_prefix="cm_", font_size=14):
        """
        각 head별 confusion matrix를 plot하고 파일로 저장.
        Args:
            output_dir (str): confusion matrix plot을 저장할 디렉토리.
            file_prefix (str): 저장 파일명 앞부분.
            font_size (int): confusion matrix plot 내 텍스트의 폰트 크기.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        confusion_matrices = self.compute_confusion_matrices()
        for head, cm in confusion_matrices.items():
            fig, ax = plt.subplots(figsize=(8, 6))

            # 폰트 크기 설정
            plt.rcParams.update({'font.size': font_size})

            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)

            # 제목 폰트 크기 명시적으로 지정
            ax.set_title(f"Confusion Matrix - {head}", fontsize=font_size + 2)
            ax.set_xlabel("Predicted label", fontsize=font_size)
            ax.set_ylabel("True label", fontsize=font_size)
            ax.tick_params(axis='both', labelsize=font_size)

            save_path = os.path.join(output_dir, f"{file_prefix}{head}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved confusion matrix plot for head '{head}' at: {save_path}")

    def get_imgnames(self):
        return self.imgnames


    def __call__(self, output: Dict, batch: Dict):
        """
        Evaluate current batch.
        Args:
            output (Dict): Regression output.
            batch (Dict): Dictionary containing images and their corresponding annotations.
        """
        imgnames = batch['imgname']
        self.imgnames += imgnames
        if 'EMDB' in self.dataset: #and '3DOH50K' 
            gt_vertices = batch['vertices']
            gt_keypoints_3d = torch.matmul(self.J_regressor_24_SMPL, gt_vertices)
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_vertices = gt_vertices - gt_pelvis

            pred_vertices = output['pred_vertices']
            pred_keypoints_3d = torch.matmul(self.J_regressor_24_SMPL, pred_vertices)
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_vertices = pred_vertices - pred_pelvis
            batch_size = pred_keypoints_3d.shape[0]
            num_samples = 1
        else:
            pred_keypoints_3d = output['pred_keypoints_3d'].detach()
            pred_keypoints_3d = pred_keypoints_3d[:,None,:,:]
            batch_size = pred_keypoints_3d.shape[0]
            num_samples = pred_keypoints_3d.shape[1]
            gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)
            gt_vertices = batch['vertices'][:,None,:,:]

            # Align predictions and ground truth such that the pelvis location is at the origin
            pred_pelvis = pred_keypoints_3d[:, :, [self.pelvis_ind]]
            gt_pelvis = gt_keypoints_3d[:, :, [self.pelvis_ind]]
            pred_keypoints_3d -= pred_pelvis
            gt_keypoints_3d -= gt_pelvis
            pred_keypoints_3d = pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3)
            gt_keypoints_3d =  gt_keypoints_3d.reshape(batch_size * num_samples, -1 ,3)
            pred_vertices = output['pred_vertices'][:,None,:,:]
            gt_vertices = gt_vertices - gt_pelvis
            pred_vertices = pred_vertices - pred_pelvis

        mpjpe, re = eval_pose(pred_keypoints_3d[:, self.keypoint_list],gt_keypoints_3d[:, self.keypoint_list])

        if hasattr(self, 'mode_pve'):
            pve = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * 1000.
            pve = pve.reshape(batch_size, num_samples)
        
        mpjpe = mpjpe.reshape(batch_size, num_samples)
        re = re.reshape(batch_size, num_samples)

        # # 예시: 분류(classification) 관련 결과 처리
        if 'per_head_cls' in output:
            for head, head_data in output['per_head_cls'].items():
                # logits와 labels detach 후 numpy 변환
                head_logits = head_data['logits'].detach()       # shape: (N_head, num_classes)
                head_labels = head_data['labels'].detach()         # shape: (N_head,)
                head_pred_labels = torch.argmax(head_logits, dim=1)
                head_labels_np = head_labels.cpu().numpy()
                head_pred_labels_np = head_pred_labels.cpu().numpy()
                # print(head_labels_np, head_pred_labels_np)
                # 분류 metric 계산
                head_accuracy = accuracy_score(head_labels_np, head_pred_labels_np)
                head_f1 = f1_score(head_labels_np, head_pred_labels_np, average='weighted')
                head_precision = precision_score(head_labels_np, head_pred_labels_np, average='weighted', zero_division=0)
                head_recall = recall_score(head_labels_np, head_pred_labels_np, average='weighted', zero_division=0)

                # 누적 저장: accuracy와 f1
                if head not in self.per_head_accuracy:
                    self.per_head_accuracy[head] = []
                    self.per_head_f1[head] = []
                self.per_head_accuracy[head].append(head_accuracy)
                self.per_head_f1[head].append(head_f1)

                # 예측값과 정답값도 저장 (confusion matrix 계산용)
                if head not in self.per_head_preds:
                    self.per_head_preds[head] = []
                    self.per_head_labels[head] = []
                self.per_head_preds[head].append(head_pred_labels_np)
                self.per_head_labels[head].append(head_labels_np)
        
                if head not in self.per_head_precision:
                    self.per_head_precision[head] = []
                    self.per_head_recall[head] = []
                self.per_head_precision[head].append(head_precision)
                self.per_head_recall[head].append(head_recall)
        
        self.counter += batch_size
        # The 0-th sample always corresponds to the mode
        if hasattr(self, 'mode_mpjpe'):
            mode_mpjpe = mpjpe[:, 0]
            self.mode_mpjpe[self.counter:self.counter+batch_size] = mode_mpjpe
        if hasattr(self, 'mode_re'):
            mode_re = re[:, 0]
            self.mode_re[self.counter:self.counter+batch_size] = mode_re
        if hasattr(self, 'mode_pve'):
            mode_pve = pve[:, 0]
            self.mode_pve[self.counter:self.counter+batch_size] = mode_pve

        self.counter += batch_size

        if hasattr(self, 'mode_mpjpe') and hasattr(self, 'mode_re'):
            return {
                'mode_mpjpe': mode_mpjpe,
                'mode_re': mode_re,
            }
        if hasattr(self, 'mode_mpjpe') and hasattr(self, 'mode_re') and hasattr(self, 'mode_pve'):
            return {
                'mode_mpjpe': mode_mpjpe,
                'mode_re': mode_re,
                'mode_pve': mode_pve,
            }
        