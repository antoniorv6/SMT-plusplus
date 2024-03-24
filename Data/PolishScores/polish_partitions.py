import os
from sklearn.model_selection import train_test_split, KFold

def partitions_by_excerpt():
    authors = os.listdir("polish_corpus_v0_dataset")
    os.makedirs("partitions_polishscores/excerpts", exist_ok=True)
    excerpts = []
    for author in authors:
        excerpts.append(f"polish_corpus_v0_dataset/{author}")
    
    kf = KFold(n_splits=5)

    fold = 0

    for train_indices, test_indices in kf.split(excerpts):
        os.makedirs(f"partitions_polishscores/excerpts/fold_{fold}", exist_ok=True)
        train_data = [excerpts[i] for i in train_indices]
        test_data = [excerpts[i] for i in test_indices]
        train_data, val_data = train_test_split(train_data,test_size=0.15)

        with open(f"partitions_polishscores/excerpts/fold_{fold}/train.txt", "w") as trainfile:
            for path in train_data:
                for file in os.listdir(path):
                    if file.endswith(".ekrn"):
                        trainfile.write(f"{path}/{file}\n")
        
        with open(f"partitions_polishscores/excerpts/fold_{fold}/val.txt", "w") as valfile:
            for path in val_data:
                for file in os.listdir(path):
                    if file.endswith(".ekrn"):
                        valfile.write(f"{path}/{file}\n")

        with open(f"partitions_polishscores/excerpts/fold_{fold}/test.txt", "w") as testfile:
            for path in test_data:
                for file in os.listdir(path):
                    if file.endswith(".ekrn"):
                        testfile.write(f"{path}/{file}\n")
        
        fold += 1

def main():
    partitions_by_excerpt()

if __name__ == "__main__":
    main()