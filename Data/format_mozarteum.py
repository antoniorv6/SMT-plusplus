import os
import json
from shutil import copy2

os.makedirs('Mozarteum', exist_ok=True)
os.makedirs('Mozarteum/mozarteum_dataset', exist_ok=True)

for folder in os.listdir('Mozarteum_v2'):
    with open(os.path.join('Mozarteum_v2', folder, 'metadata.json'), 'r') as jsonfile:
        metadata = json.load(jsonfile)
        if metadata['genre'] == "Piano":
            for image in os.listdir(os.path.join('Mozarteum_v2', folder, 'Images')):
                if image.endswith('.jpg'):
                    copy2(os.path.join('Mozarteum_v2', folder, 'Images', image), os.path.join('Mozarteum/mozarteum_dataset', image))
                    copy2(os.path.join('Mozarteum_v2', folder, 'Transcripts', image.replace('.jpg', '.krn')), 
                          os.path.join('Mozarteum/mozarteum_dataset', image.replace('.jpg', '.krn')))