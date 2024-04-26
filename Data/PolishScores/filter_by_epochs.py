import os
import json

epoch_dict = {}
root_folder = "polish_corpus_v0_dataset"
#Open a JSON file and load it
with open('polish_index.json') as f:
    data = json.load(f)
    for sample in data["data"]:
        epoch_dict[root_folder + "/" + "/".join(sample["path"].split("/")[3:])] = (sample["publication_date"] // 100) + 1

os.makedirs("polish_by_period", exist_ok=True)

for folder in os.listdir(root_folder):
    for krnfile in os.listdir(os.path.join(root_folder, folder)):
        path = os.path.join(root_folder, folder, krnfile)
        if path in epoch_dict:
            if not os.path.exists(f"polish_by_period/{epoch_dict[path]}"):
                os.makedirs(f"polish_by_period/{epoch_dict[path]}", exist_ok=True)
            #Copy the path file into the new path
            print('_'.join(path.split('/')[1:]))
            os.system(f"cp {path} polish_by_period/{epoch_dict[path]}/{'_'.join(path.split('/')[1:])}")
            os.system(f"cp {path.replace('ekrn', 'jpg')} polish_by_period/{epoch_dict[path]}/{'_'.join(path.split('/')[1:]).replace('ekrn', 'jpg')}")