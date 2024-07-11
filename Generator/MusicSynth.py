import re
import cv2
import joblib
import verovio
import random

import numpy as np
import xml.etree.ElementTree as ET

from rich import progress
from cairosvg import svg2png


memory = joblib.memory.Memory("./joblib_cache", mmap_mode="r", verbose=0)

def clean_kern(krn, avoid_tokens=['*Xped', '*tremolo', '*ped', '*Xtuplet', '*tuplet', "*Xtremolo", '*cue', '*Xcue', '*rscale:1/2', '*rscale:1', '*kcancel', '*below']):
    krn = krn.split('\n')
    newkrn = []
    # Remove the lines that contain the avoid tokens
    for idx, line in enumerate(krn):
        if not any([token in line.split('\t') for token in avoid_tokens]):
            #If all the tokens of the line are not '*'
            if not all([token == '*' for token in line.split('\t')]):
                newkrn.append(line.replace("\n", ""))
                
    return "\n".join(newkrn)

@memory.cache
def load_kern_file(path: str) -> str:
    with open(path, 'r') as file:
        krn = file.read()
        krn = clean_kern(krn)
        krn = krn.replace(" ", " <s> ")
        krn = krn.replace("\t", " <t> ")
        krn = krn.replace("\n", " <b> ")
        krn = krn.replace("·/", "")
        krn = krn.replace("·\\", "")
            
        krn = krn.split(" ")[4:]
        krn = [re.sub(r'(?<=\=)\d+', '', token) for token in krn]
        
        return " ".join(krn)

def load_from_files_list(file_ref: list, base_folder:str) -> list:
    with open(file_ref, 'r') as file:
        files = [line for line in file.read().split('\n') if line != ""]
        return [load_kern_file(base_folder + file) for file in progress.track(files)]

def rfloat(start, end):
    return round(random.uniform(start, end), 2)

def rint(start, end):
    return random.randint(start, end)        

class VerovioGenerator():
    def __init__(self, sources: list, base_folder:str, tokenization_mode='bekern'):
        self.beat_db = self.load_beats(sources, base_folder)
        verovio.enableLog(verovio.LOG_OFF)
        self.tk = verovio.toolkit()
        
        self.tokenization_mode = tokenization_mode
        
    def load_beats(self, sources: list, base_folder:str):
        sequences = load_from_files_list(sources, base_folder)
        beats = {}
        for sequence in sequences:
            lines_sequence = sequence.replace('<b>', '\n').split('\n')
            for line in lines_sequence:
                if "*M" in line:
                    beat_marker = re.search(r'\*M\S*', line)
                    beats.setdefault(beat_marker.group(), []).append(sequence)
        
        #Remove all the keys and elements that have less than 7 elements
        keys = list(beats.keys())
        for key in keys:
            if len(beats[key]) < 7:
                del beats[key]
            
        return beats

    def count_class_occurrences(self, svg_file, class_name):
        root = ET.fromstring(svg_file)
        count = 0
        # Define the namespace for SVG, if necessary
        # svg_namespace = '{http://www.w3.org/2000/svg}'
        # Traverse the SVG tree to count occurrences of the class
        for element in root.iter():
            # You can use the following line if your SVG uses a namespace:
            # if class_name in element.get('class', '').split():
            if class_name in element.get('class', ''):
                count += 1

        return count
    
    def find_image_cut(self, sample):
        # Get the height of the image
        height, _ = sample.shape[:2]

        # Iterate through the image array from the bottom to the top
        for y in range(height - 1, -1, -1):
            if [0, 0, 0] in sample[y]:
                return y

        # If no black pixel is found, return None
        return None
    
    def render(self, music_sequence):
        self.tk.loadData(music_sequence)
        self.tk.setOptions({"pageWidth": 2100, "footer": 'none', 
                                'barLineWidth': rfloat(0.3, 0.8), 'beamMaxSlope': rfloat(10,20), 
                                'staffLineWidth': rfloat(0.1, 0.3), 'spacingStaff': rfloat(1, 12)})
        self.tk.getPageCount()
        svg = self.tk.renderToSVG()
        svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")
        return svg
    
    def convert_to_png(self, svg_file, cut=False):
        pngfile = svg2png(bytestring=svg_file, background_color='white')
        pngfile = cv2.imdecode(np.frombuffer(pngfile, np.uint8), -1)
        if cut:
            cut_height = self.find_image_cut(pngfile)
            pngfile = pngfile[:cut_height + 10, :]
        
        return pngfile
    
    def generate_music_system_image(self, reduce_ratio=0.5):
        num_systems = 0
        
        while num_systems != 1:
            beat = random.choice(list(self.beat_db.keys()))
            music_seq = random.choice(self.beat_db[beat])
            render_sequence = "**kern\t**kern\n" + music_seq.replace(' <b> ', '\n').replace(' <s> ', ' ').replace(' <t> ', '\t').replace('@', '').replace('·', '')
            image = self.render(render_sequence)
            num_systems = self.count_class_occurrences(svg_file=image, class_name='grpSym')
        
        x = self.convert_to_png(image, cut=True)
        x = cv2.cvtColor(np.array(x), cv2.COLOR_BGR2RGB)
        width = int(np.ceil(x.shape[1] * reduce_ratio))
        height = int(np.ceil(x.shape[0] * reduce_ratio))
        x = cv2.resize(x, (width, height))
        
        gt_sequence = ""
        
        if self.tokenization_mode == "ekern":
            gt_sequence = "".join(music_seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", "").replace('@', '').split(" ")
        
        if self.tokenization_mode == "ekern":
            gt_sequence = "".join(music_seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace('@', '').split(" ")
        
        if self.tokenization_mode == "bekern":
            gt_sequence = "".join(music_seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")

        return x, ['<bos>'] + [token for token in gt_sequence if token != ''] + ['<eos>']

    def generate_full_page_score(self):
        raise NotImplementedError