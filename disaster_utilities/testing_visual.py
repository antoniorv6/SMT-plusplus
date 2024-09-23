import os
import cv2
import shutil
import gradio as gr
from difflib import Differ
import torch


def clean_kern(krn, avoid_tokens=['*tremolo','*staff2', '*staff1','*Xped', '*tremolo', '*ped', '*Xtuplet', '*tuplet', "*Xtremolo", '*cue', '*Xcue', '*rscale:1/2', '*rscale:1', '*kcancel', '*below']):
    krn = krn.split('\n')
    newkrn = []
    # Remove the lines that contain the avoid tokens
    for idx, line in enumerate(krn):
        if not any([token in line.split('\t') for token in avoid_tokens]):
            #If all the tokens of the line are not '*'
            if not all([token == '*' for token in line.split('\t')]):
                newkrn.append(line.replace("\n", ""))
                
    return "\n".join(newkrn)

def preprocess_kern(krn):
    krn = clean_kern(krn)
    krn = krn.replace(" ", " <s> ")
    krn = krn.replace(" /", "")
    krn = krn.replace(" \\", "")
    krn = krn.replace("·/", "")
    krn = krn.replace("·\\", "")
    krn = krn.replace("·", "").replace('@', '')
    return krn 

with open("demo_temp/prediction.ekern") as krnfile:
    prediction = preprocess_kern(krnfile.read())

def predict_seq(gt):
    d = Differ()
    print([token for token in d.compare(prediction, gt)])
    return [ (token[2:], token[0] if token[0] != " " else None) for token in d.compare(prediction, gt)]

def get_paths(path, base_folder):
    with open(f"{path}test.txt", 'r') as file:
        return [f'{base_folder}{filename.replace(".ekern", ".jpg")}' for filename in file.readlines()]

def select_image(image_path):
    image = cv2.imread(image_path)
    with open(image_path.replace(".jpg", ".ekern"), 'r') as file:
        ground_truth = preprocess_kern(file.read())


    return [image, ground_truth]

if __name__ == "__main__":
    image_paths = get_paths(path="Data/Mozarteum/partitions_mozarteum/excerpts/fold_0/", base_folder="Data/Mozarteum/")
    if not os.path.isdir("demo_temp/Mozarteum/"):
        os.makedirs("demo_temp/Mozarteum/")
        for i, image_path in enumerate(image_paths):
            shutil.copy2(image_path.replace("\n", ""), f"demo_temp/Mozarteum/")
            shutil.copy2(image_path.replace(".jpg", ".ekern").replace("\n", ""), f"demo_temp/Mozarteum/")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            explorer = gr.FileExplorer(
                root_dir="demo_temp/",
                ignore_glob="*.ekern",
                file_count="single",
                height=300,
                
            )
        with gr.Row():
            image_vis = gr.Image(label="Input image")
            ground_truth = gr.Textbox(label="Ground Truth")
            highlighted = gr.Highlightedtext(label="Diffs")
        with gr.Row():
            button = gr.Button("Transcribe")
        
        button.click(predict_seq, inputs=[ground_truth], outputs=[highlighted])
        
        explorer.change(select_image, inputs=explorer, outputs=[image_vis, ground_truth])
    
    demo.launch()