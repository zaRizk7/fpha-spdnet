import logging
import os
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from jsonargparse import CLI
from sklearn.covariance import EmpiricalCovariance as EmpCov
from sklearn.covariance import LedoitWolf as LW
from sklearn.preprocessing import StandardScaler

# Configure logger
logger = logging.getLogger("hand_pose_pipeline")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def load_hand_pose_covariances(
    root_dir: str,
    estimator: str = "lw",
    standardize: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load skeleton time series and compute connectivity matrices.

    Each recording is expected in the structure:
    root_dir/subject/label_name/recording/skeleton.txt

    Args:
        root_dir: Root directory containing hand pose dataset.
        estimator: Covariance estimator to use ('lw' for Ledoit-Wolf, 'emp' for empirical).
            'lw' is recommended for better stability.
        standardize: If True, applies StandardScaler (zero mean). This transforms
            covariance matrices into correlation matrices.
        verbose: If True, show log output.

    Returns:
        x: Covariance or correlation matrices of shape (n_samples, n_joints, n_joints)
        metadata: DataFrame with ['subject', 'recording', 'label_name']
    """
    if not verbose:
        logger.setLevel(logging.WARNING)

    scaler = StandardScaler(with_mean=False)
    cov_estimator = (
        LW(store_precision=False)
        if estimator == "lw"
        else EmpCov(store_precision=False)
    )

    subjects, recordings, label_names, cov_matrices = [], [], [], []
    total_found, total_missing = 0, 0

    for subject in sorted(os.listdir(root_dir)):
        subject_path = os.path.join(root_dir, subject)
        for label_name in sorted(os.listdir(subject_path)):
            label_path = os.path.join(subject_path, label_name)
            for recording in sorted(os.listdir(label_path)):
                fpath = os.path.join(label_path, recording, "skeleton.txt")
                try:
                    mat = np.loadtxt(fpath)[:, 1:]
                    if standardize:
                        mat = scaler.fit_transform(mat)
                    cov = cov_estimator.fit(mat).covariance_
                    # Ensure SPD for empirical covariance
                    if estimator == "emp":
                        cov += 1e-3 * np.trace(cov) * np.eye(cov.shape[0])

                    subjects.append(subject)
                    recordings.append(recording)
                    label_names.append(label_name)
                    cov_matrices.append(cov)
                    total_found += 1
                except (FileNotFoundError, IndexError, ValueError):
                    total_missing += 1

    x = np.stack(cov_matrices)
    metadata = pd.DataFrame(
        {"subject": subjects, "recording": recordings, "label_name": label_names}
    )

    logger.info(f"Found: {total_found} | Missing: {total_missing}")
    return x, metadata


def parse_train_split(
    metadata: pd.DataFrame,
    split_file: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parse split file and assign label indices and train/test flags.

    Args:
        metadata: DataFrame with ['subject', 'recording', 'label_name']
        split_file: Path to split definition file.
        verbose: If True, show log output.

    Returns:
        metadata: Updated DataFrame with 'is_train' and 'label_idx' columns.
    """
    if not verbose:
        logger.setLevel(logging.WARNING)

    is_train = np.zeros(len(metadata), dtype=bool)
    label_idx = np.full(len(metadata), -1, dtype=int)
    current_split = None

    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Train") or line.startswith("Test"):
                current_split = line.split()[0]
                continue

            if not line.startswith("Subject"):
                continue

            try:
                path, index = line.split(" ")
                subject, label_name, recording = path.split("/")
                index = int(index)
            except ValueError:
                continue

            match = (
                (metadata["subject"] == subject)
                & (metadata["label_name"] == label_name)
                & (metadata["recording"] == recording)
            )

            label_idx[match] = index
            is_train[match] = "Train" in current_split

    metadata["is_train"] = is_train
    metadata["label_idx"] = label_idx

    logger.info(f"{np.sum(is_train)} training samples found.")
    logger.info(f"{np.sum(~is_train)} test samples found.")
    logger.info(
        f"{np.unique(label_idx[label_idx >= 0]).size} unique label indices assigned."
    )

    return metadata


def assign_lexical_label_indices(
    metadata: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """
    Assign label indices based on sorted lexical order of label names.

    Args:
        metadata: DataFrame with a 'label_name' column.
        verbose: If True, log output.

    Returns:
        metadata: Updated DataFrame with 'label_idx' column.
    """
    label_order = sorted(metadata["label_name"].unique())
    label_to_index = {name: i for i, name in enumerate(label_order)}
    metadata["label_idx"] = metadata["label_name"].map(label_to_index)

    if verbose:
        logger.info("Assigned label indices based on lexical order:")
        for label, idx in label_to_index.items():
            logger.info(f"  {label}: {idx}")

    return metadata


def save_covariance_data_to_hdf5(
    filepath: str,
    x: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    """
    Save covariance/correlation matrices and metadata to HDF5.

    Args:
        filepath: Output file path (.h5)
        x: Covariance or correlation matrices
        metadata: DataFrame containing sample metadata
    """
    with h5py.File(filepath, "w") as f:
        f.create_dataset("x", data=x, compression="gzip")

        meta_group = f.create_group("metadata")
        for col in metadata.columns:
            meta_group.create_dataset(
                col,
                data=metadata[col].astype(str).to_numpy(),
                dtype=h5py.string_dtype("utf-8"),
            )

    logger.info(f"Saved dataset to {filepath}")


def main(
    root_dir: str,
    split_file: Optional[str] = None,
    output_file: str = "hand_pose_data",
    estimator: str = "lw",
    standardize: bool = True,
    verbose: bool = True,
):
    """
    Full pipeline for extracting covariance/correlation matrices and metadata from FPHA dataset.

    Args:
        root_dir: Directory containing the hand pose annotation dataset.
        split_file: Optional path to the train/test split file.
        output_file: Output file path ('.h5' extension will be added if missing).
        estimator: Covariance estimator to use ('lw' for Ledoit-Wolf, 'emp' for empirical).
            'lw' is recommended for better stability.
        standardize: If True, convert to correlation matrices by standardizing time series.
        verbose: Print progress info.
    """
    # Ensure extension is .h5
    if not output_file.endswith(".h5"):
        output_file += ".h5"

    # Make output directory if needed
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    x, metadata = load_hand_pose_covariances(root_dir, estimator, standardize, verbose)

    if split_file and os.path.exists(split_file):
        metadata = parse_train_split(metadata, split_file, verbose)
    else:
        if verbose:
            logger.warning(
                "Split file not provided or not found. Skipping train/test split."
            )
        metadata = assign_lexical_label_indices(metadata, verbose)

    save_covariance_data_to_hdf5(output_file, x, metadata)


CLI(main)
