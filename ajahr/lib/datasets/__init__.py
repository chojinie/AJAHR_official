from typing import Dict, Optional
import os, sys
            
import torch
from torch.utils.data import ConcatDataset
import numpy as np
import pytorch_lightning as pl
from yacs.config import CfgNode

import webdataset as wds
from ..configs import to_lower, dataset_eval_config
from .dataset import Dataset
from .image_dataset import ImageDataset
from .mocap_dataset import MoCapDataset
from .bedlam_dataset_tar import BedlamDataset
from .emdb_dataset import EMDBDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

def create_dataset(cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True, **kwargs) -> Dataset:
    """
    Instantiate a dataset from a config file.
    Args:
        cfg (CfgNode): Model configuration file.
        dataset_cfg (CfgNode): Dataset configuration info.
        train (bool): Variable to select between train and val datasets.
    """
    dataset_type = Dataset.registry[dataset_cfg.TYPE]
    return dataset_type(cfg, **to_lower(dataset_cfg), train=train, **kwargs)

def create_webdataset(cfg: CfgNode, dataset_cfg: CfgNode, dataset_cfg_extra: CfgNode, train: bool = True, dataset_name=None) -> Dataset:
    """
    Like `create_dataset` but load data from tars.
    """
    dataset_type = Dataset.registry[dataset_cfg.TYPE]
    return dataset_type.load_tars_as_webdataset(cfg, **to_lower(dataset_cfg), **to_lower(dataset_cfg_extra), train=train, dataset_name=dataset_name)

class MixedWebDataset(wds.WebDataset):
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> None:
        super(wds.WebDataset, self).__init__()
        dataset_list = cfg.DATASETS.TRAIN.TAR if train else cfg.DATASETS.VAL.TAR
        datasets = [create_webdataset(cfg, dataset_cfg[dataset], dataset_list[dataset], train=train, dataset_name=dataset) for dataset, v in dataset_list.items()]
        weights = np.array([v.WEIGHT for dataset, v in dataset_list.items()])
        weights = weights / weights.sum()  # normalize
        self.append(wds.RandomMix(datasets, weights))

class MixedDataset:
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True, ratio: float = 1.0) -> None:
        """
        Args:
            cfg: 설정 파일
            dataset_cfg: 개별 데이터셋 설정
            train: train/val 구분
            ratio: 전체 데이터셋 중 사용할 비율 (예: 0.3이면 30% 사용)
        """
        dataset_list = cfg.DATASETS.TRAIN.NON_TAR if train else cfg.DATASETS.VAL.NON_TAR
        self.datasets = []
        for dataset, v in dataset_list.items():
            print(f"=> Loading Dataset {dataset} (Weight: {v.WEIGHT})")
            full_dataset = create_dataset(cfg, dataset_cfg[dataset], train=train)

            if ratio < 1.0:
                num_samples = int(len(full_dataset) * ratio)
                indices = np.random.choice(len(full_dataset), num_samples, replace=False)
                full_dataset = Subset(full_dataset, indices)

            self.datasets.append(full_dataset)

        self.weights = np.array([v.WEIGHT for dataset, v in dataset_list.items()]).cumsum()

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        p = torch.rand(1).item()
        for i in range(len(self.datasets)):
            if p <= self.weights[i]:
                idx = torch.randint(0, len(self.datasets[i]), (1,)).item()
                return self.datasets[i][idx]

class TokenHMRDataModule(pl.LightningDataModule):
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for TokenHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset_nontar = None
        self.val_dataset = None
        self.train_dataset_tar = None
        self.mocap_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load datasets necessary for training
        Args:
            stage (Optional[str]): Stage of the setup (train, val, test)
        """
        if not self.train_dataset_nontar:  # 이미 초기화되었는지 확인
            self.train_dataset_nontar = MixedDataset(self.cfg, self.dataset_cfg, train=True, ratio=1.0)
            self.val_dataset = MixedDataset(self.cfg, self.dataset_cfg, train=False)
            # self.train_dataset_tar = MixedWebDataset(self.cfg, self.dataset_cfg, train=True).with_epoch(100_000).shuffle(4000)

            self.mocap_dataset = MoCapDataset(
                dataset_file=os.path.join(
                    self.cfg.DATASETS.DATASET_DIR,
                    self.dataset_cfg[self.cfg.DATASETS.MOCAP]['DATASET_FILE']
                )
            )

    def train_dataloader(self) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Setup training data loaders with two datasets (e.g., tar-based and non-tar-based).
        """
        if not self.train_dataset_nontar:  # train_dataloader 호출 전에 setup이 실행되었는지 확인
            self.setup()

        # Non-Tar DataLoader
        train_dataloader_nontar = torch.utils.data.DataLoader(
            self.train_dataset_nontar,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            drop_last=True,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            shuffle=True,
            collate_fn=self.filter_none_collate
        )

        # Tar DataLoader
        # train_dataloader_tar = torch.utils.data.DataLoader(
        #     self.train_dataset_tar,
        #     batch_size=self.cfg.TRAIN.BATCH_SIZE,
        #     drop_last=True,
        #     num_workers=self.cfg.GENERAL.NUM_WORKERS,
        #     shuffle=False,
        #     collate_fn=self.filter_none_collate
        # )

        # Return as dictionary
        # return {"nontar": train_dataloader_nontar}

        return {"nontar": train_dataloader_nontar}

    def val_dataloader(self) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Setup validation data loaders with two datasets (e.g., tar-based and non-tar-based).
        """
        if not self.val_dataloader:  # val_dataloader 호출 전에 setup이 실행되었는지 확인
            self.setup()

        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            drop_last=True,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            collate_fn=self.filter_none_collate
        )
        
        # Return as dictionary
        return val_dataloader

    def filter_none_collate(self, batch):
        """
        Custom collate function to filter out None samples from the batch.
        """
        # Filter out None values
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            raise ValueError("Filtered out all None samples; check your dataset or filtering logic.")
        for sample in batch:
            if 'domain_label' not in sample:
                sample['domain_label'] = 0  # 기본값 설정
        return torch.utils.data._utils.collate.default_collate(batch)

    # def filter_none_collate(self, batch):
    #     """
    #     Custom collate function to filter out None samples from the batch.
    #     """
    #     # Filter out None values
    #     batch = [x for x in batch if x is not None]
    #     if len(batch) == 0:
    #         raise ValueError("Filtered out all None samples; check your dataset or filtering logic.")
        
    #     # 개별 요소 추출 (예: input, labels 등)
    #     labels = [torch.tensor(item['label']) for item in batch]  # 리스트를 텐서로 변환
        
    #     # 라벨을 패딩하여 같은 크기로 맞춤
    #     labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    #     return {'label': labels_padded}
