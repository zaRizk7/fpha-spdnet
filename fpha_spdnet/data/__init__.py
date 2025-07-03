import h5py
from lightning import LightningDataModule
import numpy as np
from sklearn.model_selection import check_cv
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["FPHADataset"]


class FPHADataModule(LightningDataModule):
    def __init__(self, h5_path: str, batch_size: int = 32, num_workers: int = 0):
        """
        DataModule for FPHA dataset.

        Args:
            h5_path (str): Path to the HDF5 file containing preprocessed data.
            batch_size (int): Batch size for training and validation.
        """
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = FPHADataset(self.h5_path)

    @property
    def num_classes(self) -> int:
        """
        Returns the number of unique classes in the dataset.
        """
        return self.dataset.num_classes

    @property
    def matrix_size(self) -> int:
        """
        Returns the size of the SPD matrices (i.e., matrix is of shape [d, d]).
        """
        return self.dataset.matrix_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset.get_split("train"),
            self.batch_size,
            True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        val_dataset = self.dataset.get_split("test")
        return DataLoader(
            val_dataset,
            len(val_dataset),
            False,
            num_workers=self.num_workers,
        )


class FPHADataset(Dataset):
    """
    FPHA Dataset backed by HDF5. Supports lazy loading, train/test splitting,
    and memory-efficient cross-validation splitting.

    Args:
        h5_path (str): Path to the HDF5 file containing preprocessed data.
        indices (list[int], optional): Sample indices to load. If None, use all.
    """

    def __init__(self, h5_path: str, indices: list[int] | None = None):
        self.h5_path = h5_path
        self._file = h5py.File(h5_path, "r")  # open once, close at __del__

        self.label_idx_all = self._file["metadata"]["label_idx"][:].astype(int)
        self.is_train = (
            self._file["metadata"]["is_train"][:].astype(str) == "True"
            if "is_train" in self._file["metadata"]
            else None
        )
        self._matrix_size = self._file["x"].shape[1]  # Assuming shape (n_samples, d, d)

        self.indices = np.asarray(indices) if indices is not None else np.arange(len(self.label_idx_all))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

        if isinstance(idx, (int, np.integer)):
            real_idx = self.indices[idx]
            x = torch.tensor(self._file["x"][real_idx], dtype=torch.float32)
            y = int(self.label_idx_all[real_idx])
            return x, y

        # Handle array-like batch indexing
        indices = np.array(self.indices)[np.array(idx)]
        sort_order = np.argsort(indices)
        sorted_indices = indices[sort_order]

        x_sorted = self._file["x"][sorted_indices]
        y_sorted = self.label_idx_all[sorted_indices]

        restore_order = np.argsort(sort_order)
        x = torch.tensor(x_sorted[restore_order], dtype=torch.float32)
        y = torch.tensor(y_sorted[restore_order], dtype=torch.long)

        return x, y

    def get_split(self, split: str) -> "FPHADataset":
        """
        Return train or test split using metadata["is_train"].

        Args:
            split (str): One of {"train", "test"}.

        Returns:
            FPHADataset: Filtered dataset.
        """
        assert split in {"train", "test"}, "Split must be 'train' or 'test'"
        if self.is_train is None:
            raise ValueError("Split information is missing. Was split_file used?")
        mask = self.is_train if split == "train" else ~self.is_train
        split_indices = np.where(mask)[0]
        return FPHADataset(self.h5_path, indices=split_indices)

    def split_cv(self, cv):
        """
        Generator yielding (train_dataset, val_dataset) splits using scikit-learn CV.

        Args:
            cv (int or sklearn.model_selection.BaseCrossValidator): Either number of folds or CV object.

        Yields:
            Tuple[FPHADataset, FPHADataset]: Train and validation datasets for each fold.
        """
        label_subset = self.label_idx_all[self.indices]
        all_indices = self.indices

        cv_obj = check_cv(cv, y=label_subset, classifier=True)

        for train_idx, val_idx in cv_obj.split(all_indices, label_subset):
            yield (
                FPHADataset(self.h5_path, indices=all_indices[train_idx]),
                FPHADataset(self.h5_path, indices=all_indices[val_idx]),
            )

    @property
    def num_classes(self) -> int:
        """
        Returns the number of unique classes in the current dataset.
        """
        return len(np.unique(self.label_idx_all[self.indices]))

    @property
    def matrix_size(self) -> int:
        """
        Returns the size of the SPD matrices (i.e., matrix is of shape [d, d]).
        """
        return self._matrix_size

    def __repr__(self):
        split_info = ""
        if self.is_train is not None:
            if np.all(self.is_train[self.indices]):
                split_info = " (train split)"
            elif np.all(~self.is_train[self.indices]):
                split_info = " (test split)"
            else:
                split_info = " (mixed split)"

        items = [
            f"path='{self.h5_path}'",
            f"n_samples={len(self)}",
            f"matrix_size={self.matrix_size}x{self.matrix_size}",
            f"num_classes={self.num_classes}",
        ]

        repr_str = f"{self.__class__.__name__}("
        line_len = len(repr_str)
        indent = " " * len(repr_str)

        for i, item in enumerate(items):
            if i > 0:
                repr_str += ", "
                line_len += 2
            if line_len + len(item) > 100:
                repr_str += f"\n{indent}"
                line_len = len(indent)
            repr_str += item
            line_len += len(item)

        repr_str += f"{split_info})"
        return repr_str

    def __del__(self):
        if self._file is not None:
            self._file.close()
