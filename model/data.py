import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

PATH_DATASETS = "/home/dxzielinski/Downloads/cats"
BATCH_SIZE = 256
NUM_WORKERS = 18
MIXED_DATASET = "/home/dxzielinski/Downloads/dogs-vs-cats"


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
        self.transform_train = transforms.Compose(
            [
                transforms.Resize((self.dims[1], self.dims[2])),
                transforms.RandomApply([transforms.RandomRotation(30)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomErasing(p=0.2, scale=(0.001, 0.01), ratio=(1.2, 1.8)),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.dims[1], self.dims[2])),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.seed = seed

    def setup(self, stage=None):
        full_dataset = DatasetFolder(
            root=self.data_dir,
            transform=None,
            loader=default_loader,
            extensions=("png"),
        )
        N = len(full_dataset)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(N, generator=g).tolist()
        n_train = int(0.6 * N)
        n_val = int(0.2 * N)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]
        train_ds_full = DatasetFolder(
            root=self.data_dir,
            loader=default_loader,
            extensions=("png",),
            transform=self.transform_train,
        )
        val_ds_full = DatasetFolder(
            root=self.data_dir,
            loader=default_loader,
            extensions=("png",),
            transform=self.transform,
        )
        test_ds_full = DatasetFolder(
            root=self.data_dir,
            loader=default_loader,
            extensions=("png",),
            transform=self.transform,
        )
        if stage == "fit":
            self.train_ds = Subset(train_ds_full, train_idx)
            self.val_ds = Subset(val_ds_full, val_idx)

        if stage == "test":
            self.test_ds = Subset(test_ds_full, test_idx)

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


class CatsDogsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = MIXED_DATASET,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 64, 64)
        self.transform_train = transforms.Compose(
            [
                transforms.Resize((self.dims[1], self.dims[2])),
                transforms.RandomApply([transforms.RandomRotation(30)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomErasing(p=0.2, scale=(0.001, 0.01), ratio=(1.2, 1.8)),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.dims[1], self.dims[2])),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.seed = seed

    def setup(self, stage=None):
        train_dataset = DatasetFolder(
            root=f"{self.data_dir}/train1",
            transform=None,
            loader=default_loader,
            extensions=("jpg"),
        )
        test_dataset = DatasetFolder(
            root=f"{self.data_dir}/test1",
            transform=self.transform,
            loader=default_loader,
            extensions=("jpg"),
        )
        N = len(train_dataset)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(N, generator=g).tolist()
        n_train = int(0.8 * N)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        train_ds_full = DatasetFolder(
            root=self.data_dir,
            loader=default_loader,
            extensions=("jpg",),
            transform=self.transform_train,
        )
        val_ds_full = DatasetFolder(
            root=self.data_dir,
            loader=default_loader,
            extensions=("jpg",),
            transform=self.transform,
        )
        if stage == "fit":
            self.train_ds = Subset(train_ds_full, train_idx)
            self.val_ds = Subset(val_ds_full, val_idx)

        if stage == "test":
            self.test_ds = test_dataset

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
