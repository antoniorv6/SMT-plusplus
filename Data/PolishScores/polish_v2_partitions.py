import os
from sklearn.model_selection import KFold, train_test_split

files = [file for file in os.listdir("polish_corpus_v2_dataset") if file.endswith('ekrn')]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(files)):
    os.makedirs(f"partitions_polishscores/excerpts/fold_{i}", exist_ok=True)
    train_files = ["polish_corpus_v2_dataset/" + files[i] for i in train_index]
    test_files = ["polish_corpus_v2_dataset/" + files[i] for i in test_index]
    #include validation
    train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)
    
    with open(f'partitions_polishscores/excerpts/fold_{i}/train.txt', 'w') as f:
        f.write('\n'.join(train_files))
    with open(f'partitions_polishscores/excerpts/fold_{i}/test.txt', 'w') as f:
        f.write('\n'.join(test_files))
    with open(f'partitions_polishscores/excerpts/fold_{i}/val.txt', 'w') as f:
        f.write('\n'.join(val_files))