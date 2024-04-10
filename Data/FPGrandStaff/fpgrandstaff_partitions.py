import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

def prepare_data_for_cross_validation(data):
    """
    Prepare the dataset for stratified K-Fold cross-validation.

    :param data: Dictionary with folder names as keys and lists of file paths as values.
    :return: A tuple of samples and labels ready for cross-validation.
    """
    samples = []
    labels = []
    for folder_name, file_paths in data.items():
        samples.extend(file_paths)
        labels.extend([folder_name] * len(file_paths))
    return samples, labels

def stratified_k_fold_with_validation(samples, labels, n_splits=5, validation_size=0.2, folder="polish_partitions/excerpts/"):
    """
    Perform stratified K-Fold cross-validation with a validation set.

    :param samples: List of all samples (file paths in this case).
    :param labels: List of labels corresponding to each sample.
    :param n_splits: Number of folds.
    :param validation_size: Proportion of the training set to be used as validation.
    """
    skf = StratifiedKFold(n_splits=n_splits)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(samples, labels)):
        # Splitting the data into training/testing sets for the current fold
        X_train, X_test = [samples[i] for i in train_idx], [samples[i] for i in test_idx]
        y_train, y_test = [labels[i] for i in train_idx], [labels[i] for i in test_idx]
        
        # Further split the training set to create a validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, stratify=y_train)
        
        os.makedirs(f"{folder}fold_{fold}", exist_ok=True)
        
        with open(f"{folder}fold_{fold}/train.txt", "w") as f:
            f.write("\n".join(X_train))
        
        with open(f"{folder}fold_{fold}/val.txt", "w") as f:
            f.write("\n".join(X_val))
        
        with open(f"{folder}fold_{fold}/test.txt", "w") as f:
            f.write("\n".join(X_test))
        
        print(f"Fold {fold+1}:")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Testing samples: {len(X_test)}")
        # Here, you would process your training, validation, and testing sets

def load_data_from_folders(root_dir):
    """
    Load data from subfolders and ensure each folder has enough samples.
    Each subfolder represents a different class or category.

    :param root_dir: Root directory containing subfolders with .ekrn files.
    :return: A dictionary with folder names as keys and lists of file paths as values.
    """
    data = {}
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        for subfolder in os.listdir(folder_path):
            folder_path = os.path.join(folder_path, subfolder)
            for subsubfolder in os.listdir(folder_path):
                folder_path = os.path.join(folder_path, subsubfolder)
                if os.path.isdir(folder_path):
                    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.ekrn')]
                    # Ensure each folder has a minimum number of samples, let's say at least 3 for this example
                    if len(file_paths) >= 3:
                        data[folder_name] = file_paths
    return data

# Example usage
os.makedirs("partitions_fpgrandstaff/excerpts", exist_ok=True)
root_dir = 'fp_grandstaff_dataset'
data = load_data_from_folders(root_dir)
print(data)
import sys
sys.exit()
samples, labels = prepare_data_for_cross_validation(data)
stratified_k_fold_with_validation(samples, labels, n_splits=5)
