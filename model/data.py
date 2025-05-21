import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

PATH_DATASETS = "/home/dxzielinski/Downloads/cats"
BATCH_SIZE = 256
NUM_WORKERS = 18


class CatsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dims = (3, 64, 64)
        # it's not used for now
        self.transform_train = transforms.Compose(
            [
                transforms.RandomApply(transforms.RandomRotation(30), p=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.001, 0.01), ratio=(1.2, 1.8)),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.seed = seed

    def setup(self, stage=None):
        full_dataset = DatasetFolder(
            root=self.data_dir,
            transform=self.transform,
            loader=default_loader,
            extensions=("png"),
        )
        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [0.6, 0.2, 0.2],
            generator=torch.Generator().manual_seed(self.seed),
        )

        if stage == "fit":
            self.train_ds = train_ds
            self.val_ds = val_ds

        if stage == "test":
            self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
