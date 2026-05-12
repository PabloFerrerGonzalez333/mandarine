"""Explicit datamodule for the deterministic mandarine splits."""

from __future__ import annotations

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.folder import FolderDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode
from torchvision.transforms.v2 import ColorJitter, Compose, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, Resize

from mandarine.config import DatasetConfig
from mandarine.data.splits import SplitPaths


def build_train_augmentations(image_size: int) -> Compose:
    """CPU-friendly augmentations for normal training images."""

    return Compose(
        [
            Resize((image_size, image_size), antialias=True),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.15),
            RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
        ],
    )


def build_eval_augmentations(image_size: int) -> Compose:
    """Stable evaluation transforms."""

    return Compose([Resize((image_size, image_size), antialias=True)])


class MandarineDataModule(AnomalibDataModule):
    """Datamodule backed by explicit train/val/test directories."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        split_paths: SplitPaths,
        *,
        train_batch_size: int | None = None,
        eval_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size or dataset_config.train_batch_size,
            eval_batch_size=eval_batch_size or dataset_config.eval_batch_size,
            num_workers=dataset_config.num_workers,
            train_augmentations=build_train_augmentations(dataset_config.image_size),
            val_augmentations=build_eval_augmentations(dataset_config.image_size),
            test_augmentations=build_eval_augmentations(dataset_config.image_size),
            test_split_mode=TestSplitMode.FROM_DIR,
            val_split_mode=ValSplitMode.FROM_DIR,
            seed=split_paths.seed,
        )
        self._name = dataset_config.name
        self.dataset_config = dataset_config
        self.split_paths = split_paths

    @property
    def name(self) -> str:
        return self._name

    def _setup(self, _stage: str | None = None) -> None:
        train_anchor = self.split_paths.train_good
        extensions = self.dataset_config.image_extensions

        self.train_data = FolderDataset(
            name=self.name,
            normal_dir=train_anchor,
            split=Split.TRAIN,
            extensions=extensions,
        )
        self.val_data = FolderDataset(
            name=self.name,
            normal_dir=train_anchor,
            normal_test_dir=self.split_paths.val_good,
            abnormal_dir=self.split_paths.val_bad,
            split=Split.TEST,
            extensions=extensions,
        )
        self.test_data = FolderDataset(
            name=self.name,
            normal_dir=train_anchor,
            normal_test_dir=self.split_paths.test_good,
            abnormal_dir=self.split_paths.test_bad,
            split=Split.TEST,
            extensions=extensions,
        )
