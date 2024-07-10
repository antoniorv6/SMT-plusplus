import os
from shutil import copy2
import kernpy as kp

os.makedirs('failed_conversions', exist_ok=True)
files = []

options = kp.ExportOptions(spine_types=['**kern'], token_categories=kp.BEKERN_CATEGORIES, kern_type=kp.KernTypeExporter.eKern)

for file in os.listdir('Mozarteum/mozarteum_dataset'):
    if file.endswith('.krn'):
        try:
            document, _ = kp.read(os.path.join('Mozarteum/mozarteum_dataset', file))
            kp.store(document, os.path.join('Mozarteum/mozarteum_dataset', file.replace('.krn', '.ekern')), options)
            files.append(os.path.join('Mozarteum/mozarteum_dataset', file.replace('.krn', '.ekern')))
        except:
            print(f'Error with {file}')
            copy2(os.path.join('Mozarteum/mozarteum_dataset', file), os.path.join('failed_conversions', file))

print(f'Converted {len(files)} files')