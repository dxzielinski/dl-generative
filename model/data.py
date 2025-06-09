import os
import glob
from PIL import Image
import random
import shutil
import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, ImageFolder
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


def prepare_cats_dogs_cyclic():
    """
    Reorganize a single-source dogs-vs-cats folder into a cyclic structure:
      dogs-vs-cats-cycle/
        train/dog, train/cat
        val/dog,   val/cat
        test/dog,  test/cat

    Assumes all images are under:
      /home/dxzielinski/Downloads/dogs-vs-cats/train1/train/{dog,cat}
    Splits each category by default into 70% train, 20% val, 10% test.
    """
    input_dir = "/home/dxzielinski/Downloads/dogs-vs-cats/train1/train"
    output_dir = "/home/dxzielinski/Downloads/dogs-vs-cats-cycle"
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Please remove it first.")
        return
    val_ratio = 0.2
    test_ratio = 0.1
    seed = 42
    random.seed(seed)
    all_files = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
    ]
    categories = {
        "cat": [f for f in all_files if f.lower().startswith("cat.")],
        "dog": [f for f in all_files if f.lower().startswith("dog.")],
    }
    for split in ["train", "val", "test"]:
        for category in categories:
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

    for category, images in categories.items():
        random.shuffle(images)
        total = len(images)
        n_test = int(total * test_ratio)
        n_val = int(total * val_ratio)
        test_imgs = images[:n_test]
        val_imgs = images[n_test : n_test + n_val]
        train_imgs = images[n_test + n_val :]
        for img in train_imgs:
            shutil.copy(
                os.path.join(input_dir, img),
                os.path.join(output_dir, "train", category, img),
            )
        for img in val_imgs:
            shutil.copy(
                os.path.join(input_dir, img),
                os.path.join(output_dir, "val", category, img),
            )
        for img in test_imgs:
            shutil.copy(
                os.path.join(input_dir, img),
                os.path.join(output_dir, "test", category, img),
            )
    print(f"Reorganized dataset created at: {output_dir}")


class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transforms_=None, unaligned=False):
        """
        Unpaired dataset for two domains.
        - root_A: str, e.g. ".../dogs-vs-cats-cycle/train/cat"
        - root_B: str, e.g. ".../dogs-vs-cats-cycle/train/dog"
        - transforms_: torchvision transforms (callable)
        - unaligned: if True, draw B randomly
        """
        self.transform = transforms_
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root_A, "*.*")))
        self.files_B = sorted(glob.glob(os.path.join(root_B, "*.*")))

    def __getitem__(self, index):
        # load A
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
        A = self.transform(img_A)

        # load B (aligned index or random)
        if self.unaligned:
            idx_B = random.randint(0, len(self.files_B) - 1)
        else:
            idx_B = index % len(self.files_B)
        img_B = Image.open(self.files_B[idx_B]).convert("RGB")
        B = self.transform(img_B)

        return {"A": A, "B": B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class CycleGANDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=256, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 64, 64)

        # your existing transforms
        self.transform_train = transforms.Compose(
            [
                transforms.Resize(self.dims[1:]),
                transforms.RandomApply([transforms.RandomRotation(30)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
                transforms.RandomErasing(p=0.2, scale=(0.001, 0.01), ratio=(1.2, 1.8)),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.Resize(self.dims[1:]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_ds = ImageDataset(
                root_A=os.path.join(self.data_dir, "train", "cat"),
                root_B=os.path.join(self.data_dir, "train", "dog"),
                transforms_=self.transform_train,
                unaligned=True,
            )
            self.val_ds = ImageDataset(
                root_A=os.path.join(self.data_dir, "val", "cat"),
                root_B=os.path.join(self.data_dir, "val", "dog"),
                transforms_=self.transform_test,
                unaligned=False,
            )
        if stage in (None, "test"):
            self.test_ds = ImageDataset(
                root_A=os.path.join(self.data_dir, "test", "cat"),
                root_B=os.path.join(self.data_dir, "test", "dog"),
                transforms_=self.transform_test,
                unaligned=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


if __name__ == "__main__":
    prepare_cats_dogs_cyclic()
