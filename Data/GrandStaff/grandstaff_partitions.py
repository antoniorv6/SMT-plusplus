import os
from sklearn.model_selection import train_test_split, KFold

def partitions_by_author():
    authors = os.listdir("grandstaff_dataset")
    os.makedirs("partitions_grandstaff/authors", exist_ok=True)
    test_samples = []
    train_samples = []
    for author in authors:
        os.makedirs(f"partitions_grandstaff/authors/{author}", exist_ok=True)
        for subdir in os.listdir(f"grandstaff_dataset/{author}"):
                for subsubdir in os.listdir(f"grandstaff_dataset/{author}/{subdir}"):
                    for file in os.listdir(f"grandstaff_dataset/{author}/{subdir}/{subsubdir}"):
                        if file.endswith(".krn"):
                            test_samples.append(f"grandstaff_dataset/{author}/{subdir}/{subsubdir}/{file}\n")
        for auth in authors:
            if auth != author:
                for subdir in os.listdir(f"grandstaff_dataset/{auth}"):
                    for subsubdir in os.listdir(f"grandstaff_dataset/{auth}/{subdir}"):
                        for file in os.listdir(f"grandstaff_dataset/{auth}/{subdir}/{subsubdir}"):
                            if file.endswith(".krn"):
                                train_samples.append(f"grandstaff_dataset/{auth}/{subdir}/{subsubdir}/{file}\n")
    
        train, val = train_test_split(train_samples, test_size=0.2)
    
        with open(f"partitions_grandstaff/authors/{author}/train.txt", "w") as trainfile:
            for sample in train:
                trainfile.write(sample)
        
        with open(f"partitions_grandstaff/authors/{author}/val.txt", "w") as valfile:
            for sample in val:
                valfile.write(sample)
        
        with open(f"partitions_grandstaff/authors/{author}/test.txt", "w") as testfile:
            for sample in test_samples:
                testfile.write(sample)

        test_samples = []
        train_samples = []

def partitions_by_excerpt():
    authors = os.listdir("grandstaff_dataset")
    os.makedirs("partitions_grandstaff/excerpts", exist_ok=True)
    excerpts = []
    for author in authors:
        for subdir in os.listdir(f"grandstaff_dataset/{author}"):
                for subsubdir in os.listdir(f"grandstaff_dataset/{author}/{subdir}"):
                    excerpts.append(f"grandstaff_dataset/{author}/{subdir}/{subsubdir}")
    
    kf = KFold(n_splits=5)

    fold = 0

    for train_indices, test_indices in kf.split(excerpts):
        os.makedirs(f"partitions_grandstaff/excerpts/fold_{fold}", exist_ok=True)
        train_data = [excerpts[i] for i in train_indices]
        test_data = [excerpts[i] for i in test_indices]
        train_data, val_data = train_test_split(train_data,test_size=0.2)

        with open(f"partitions_grandstaff/excerpts/fold_{fold}/train.txt", "w") as trainfile:
            for path in train_data:
                for file in os.listdir(path):
                    if file.endswith(".krn"):
                        trainfile.write(f"{path}/{file}\n")
        
        with open(f"partitions_grandstaff/excerpts/fold_{fold}/val.txt", "w") as valfile:
            for path in val_data:
                for file in os.listdir(path):
                    if file.endswith(".krn"):
                        valfile.write(f"{path}/{file}\n")

        with open(f"partitions_grandstaff/excerpts/fold_{fold}/test.txt", "w") as testfile:
            for path in test_data:
                for file in os.listdir(path):
                    if file.endswith(".krn"):
                        testfile.write(f"{path}/{file}\n")
        
        fold += 1

def partitions_by_type():
    authors = os.listdir("grandstaff_dataset")
    os.makedirs("partitions_grandstaff/types", exist_ok=True)
    original_excerpts = []
    augmented_excerpts = []
    for author in authors:
        for subdir in os.listdir(f"grandstaff_dataset/{author}"):
                for subsubdir in os.listdir(f"grandstaff_dataset/{author}/{subdir}"):
                    for file in os.listdir(f"grandstaff_dataset/{author}/{subdir}/{subsubdir}"):
                        if "original" in file and file.endswith(".ekrn"):
                            original_excerpts.append(f"grandstaff_dataset/{author}/{subdir}/{subsubdir}/{file}")
                        elif file.endswith(".ekrn"):
                            augmented_excerpts.append(f"grandstaff_dataset/{author}/{subdir}/{subsubdir}/{file}")
    
    augmented_train, augmented_val = train_test_split(augmented_excerpts, test_size=0.1)

    with open(f"partitions_grandstaff/types/train.txt", "w") as trainfile:
        for sample in augmented_train:
            trainfile.write(f"{sample}\n")

    with open(f"partitions_grandstaff/types/val.txt", "w") as valfile:
        for sample in augmented_val:
            valfile.write(f"{sample}\n")
    
    with open(f"partitions_grandstaff/types/test.txt", "w") as testfile:
        for sample in original_excerpts:
            testfile.write(f"{sample}\n")

def main():
    partitions_by_type()

if __name__ == "__main__":
    main()