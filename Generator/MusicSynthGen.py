import os
import re
import cv2
import names
import random
import verovio
import numpy as np
from wonderwords import RandomSentence
from wand.image import Image as IMG
from cairosvg import svg2png
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
from rich import progress

def simplify_tokens(tokens):
    simplified_tokens = []
    for token in tokens:
        token = re.sub(r'::+', '', token)
        simplified_tokens.append(token)
    return simplified_tokens

def rfloat(start, end):
    return round(random.uniform(start, end), 2)

def rint(start, end):
    return random.randint(start, end)

def erase_numbers_in_tokens_with_equal(tokens):
       return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

def load_data_from_krn(path, base_folder="GrandStaff", krn_type="ekrn", tokenization_mode="standard"):
    y = []
    with open(path) as datafile:
        lines = datafile.readlines()
        for line in progress.track(lines):
            excerpt = line.replace("\n", "")
            try:
                with open(f"Data/{base_folder}/{'.'.join(excerpt.split('.')[:-1])}.{krn_type}") as krnfile:
                    krn_content = krnfile.read()
                    krn = krn_content.replace(" ", " <s> ")
                    krn = krn.replace("\t", " <t> ")
                    krn = krn.replace("\n", " <b> ")
                    krn = krn.replace("\n", " <b> ")
                    
                    if tokenization_mode == "ekern":
                        krn = krn.replace("@", "")
                    if tokenization_mode == "standard":
                        krn = krn.replace("·", "")
                        krn = krn.replace("@", "")

                    krn = krn.replace("/", "")
                    krn = krn.replace("\\", "")
                    krn = krn.split(" ")
                    y.append(erase_numbers_in_tokens_with_equal(krn))
                    
            except Exception as e:
                print(f'Error reading Data/{base_folder}/{excerpt}')
                print(e)

    return y

class VerovioGenerator():
    def __init__(self, gt_samples_path, textures_path="Generator/paper_textures", fixed_number_systems=False, tokenization_method="standard") -> None:
        self.tk = verovio.toolkit()
        verovio.enableLog(verovio.LOG_OFF)
        self.fixed_systems = fixed_number_systems
        self.tokenization_method = tokenization_method
        self.beats = self.load_beats(gt_samples_path)
        self.title_generator = RandomSentence()
        self.textures = [os.path.join(textures_path, f) for f in os.listdir(textures_path) if os.path.isfile(os.path.join(textures_path, f))]
    
    def load_beats(self, path):
        """Load beats from the provided path."""
        beats = {}
        data = load_data_from_krn(path, tokenization_mode=self.tokenization_method)
        for sample in data:
            if sample.count('*-') == 2:
                beat_marker = re.search(r'\*M\S*', " ".join(sample))
                if beat_marker:
                    beat = beat_marker.group()
                    beats.setdefault(beat, []).append(sample)
        return beats

    def filter_system_continuation(self, system):
        index = 0
        # Locate where the rule metrics end in the system
        for i, item in enumerate(system):
            if item == "=-":
                index = i
                break
        
        to_filter = system[:index]
        filtered = []
        idx = 0
        while idx < len(to_filter):
            element = to_filter[idx]
            if 'clef' not in element and 'M' not in element and 'k' not in element:
                filtered.append(element)
                filtered.append(to_filter[idx+1])
            idx+=2

        return filtered + system[index:]
    
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

    def inkify_image(self, sample):
        image = IMG.from_array(np.array(sample))
        paint = rfloat(0, 1)
        image.oil_paint(paint)
        #rotate = rfloat(-1, 1)
        #image.rotate(degree=rotate, background="WHITE")
        #blur1, blur2, blur3 = rfloat(-5, 5), rfloat(-5, 5), rfloat(-5, 5)
        #image.motion_blur(radius=blur1, sigma=blur2, angle=blur3)
        
        return Image.fromarray(np.array(image))
    
    def texturize_image(self, x):
        texture = Image.open(random.choice(self.textures))
        img_width, img_height = x.shape[1], x.shape[0]
        #Check if the texture is smaller than the image. If so, resize it to the image size
        if texture.size[0] < img_width or texture.size[1] < img_height:
            texture = texture.resize((img_width, img_height))
        
        x = self.inkify_image(x)
        x = np.array(x)
        music_image = Image.fromarray(x)
        left = random.randint(0, texture.size[0] - img_width)
        top = random.randint(0, texture.size[1] - img_height)
        texture = texture.crop((left, top, left + img_width, top + img_height))
        inverted_music_image = ImageOps.invert(music_image.convert('RGB'))
        mask = inverted_music_image.convert("L")
        texture.paste(music_image, mask=mask)
        x = texture#cv2.cvtColor(np.array(texture), cv2.COLOR_RGB2BGR)
        
        return x

    def generate_system(self, reduce_ratio=0.5, add_texture=False):
        
        num_sys_gen = 1
        
        generated_systems = np.inf
        while generated_systems != num_sys_gen:
            length = 0
            while length < num_sys_gen:
                beat = random.choice(list(self.beats.keys()))
                systems = self.beats[beat]
                length = len(systems)
            
            sequence = random.choice(systems)
            krnseq = "".join(sequence[:-1]).replace("@", "").replace("<s>", " ").replace("<b>", "\n").replace("<t>", "\t").replace("**ekern", "**kern")
            
            self.tk.loadData(krnseq)
            self.tk.setOptions({"pageWidth": 2100, "footer": 'none', 
                                'barLineWidth': rfloat(0.3, 0.8), 'beamMaxSlope': rfloat(10,20), 
                                'staffLineWidth': rfloat(0.1, 0.3), 'spacingStaff': rfloat(1, 12)})
            
            self.tk.getPageCount()
            svg = self.tk.renderToSVG()
            svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")

            generated_systems = self.count_class_occurrences(svg_file=svg, class_name='grpSym')
        
        pngfile = svg2png(bytestring=svg, background_color='white')
        pngfile = cv2.imdecode(np.frombuffer(pngfile, np.uint8), -1)

        height = self.find_image_cut(pngfile)
        pngfile = pngfile[:height + 10, :]

        x = pngfile

        if add_texture == True:
            x = self.texturize_image(x)
        else:
            x = np.array(x)
        
        x = cv2.cvtColor(np.array(x), cv2.COLOR_BGR2RGB)
        
        width = int(np.ceil(pngfile.shape[1] * reduce_ratio))
        height = int(np.ceil(pngfile.shape[0] * reduce_ratio))
        x = cv2.resize(x, (width, height))

        if self.tokenization_method == "ekern":
            sequence = "".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").split(" ")
        
        if self.tokenization_method == "bekern":
            sequence = "".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")

        return x, ['<bos>'] + sequence[4:-1] + ['<eos>']


    def generate_score(self, num_sys_gen=1, padding=10, 
                       reduce_ratio=0.5, random_margins=True, check_generated_systems=True, 
                       cut_height=True, add_texture=False, 
                       include_title=False, include_author=False, page_size=None):
        
        n_sys_generate = random.randint(1, num_sys_gen)
        if self.fixed_systems:
            n_sys_generate = num_sys_gen
        
        margins = None

        if random_margins:
            margins = [rint(25, 200) for _ in range(4)]
        
        generated_systems = np.inf
        while generated_systems != n_sys_generate:
            length = 0

            while length < num_sys_gen:
                beat = random.choice(list(self.beats.keys()))
                systems = self.beats[beat]
                length = len(systems)

            random_systems = random.sample(systems, n_sys_generate)

            sequence = []

            if n_sys_generate > 1:
                first_seq = "".join(random_systems[0]).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").split(" ")
                sequence = ['=' if token == '==' else token for token in first_seq[:-5]]
                for seq in random_systems[1:-1]:
                    seq = "".join(seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").split(" ")
                    sequence += self.filter_system_continuation(seq[4:-5])
                
                last_seq = "".join(random_systems[-1]).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").split(" ")
                sequence += self.filter_system_continuation(last_seq[4:])
            else:
                sequence = random_systems[0]
            
            preseq = ""
            preseq += f"!!!OTL:{self.title_generator.sentence()}\n" if include_title else ""
            preseq += f"!!!COM:{names.get_full_name()}\n" if include_author else ""
            
            krnseq = preseq + "".join(sequence[:-1]).replace("@", "").replace("<s>", " ").replace("<b>", "\n").replace("<t>", "\t").replace("**ekern", "**kern")

            #with open("test.krn", "w") as krnfile:
            #    krnfile.write(krnseq)
            #with open("init.krn", "w") as krnfile:
            #    krnfile.write("".join(random_systems[0]).replace("<s>", " ").replace("<b>", "\n").replace("<t>", "\t").replace("**ekern_1.0", "**kern"))
            #with open("end.krn", "w") as krnfile:
            #    krnfile.write("".join(random_systems[-1]).replace("<s>", " ").replace("<b>", "\n").replace("<t>", "\t").replace("**ekern_1.0", "**kern"))   
            
            self.tk.loadData(krnseq)
            
            if page_size != None:
                self.tk.setOptions({"pageWidth": page_size[1], "pageHeight": page_size[0], "footer": 'none', 'barLineWidth': rfloat(0.3, 0.8), 'beamMaxSlope': rfloat(10,20), 'staffLineWidth': rfloat(0.1, 0.3), 'spacingStaff': rfloat(1, 12)})
            #if random_margins:
            #    self.tk.setOptions({"pageWidth": 2100, "pageMarginLeft":margins[0], "pageMarginRight":margins[1], "pageMarginTop":margins[2], "pageMarginBottom":margins[3], 
            #                    "footer": 'none', 'barLineWidth': rfloat(0.3, 0.8), 'beamMaxSlope': rfloat(10,20), 'staffLineWidth': rfloat(0.1, 0.3), 'spacingStaff': rfloat(1, 12)})
            else:
                self.tk.setOptions({"pageWidth": 2100, "footer": 'none', 'barLineWidth': rfloat(0.3, 0.8), 'beamMaxSlope': rfloat(10,20), 'staffLineWidth': rfloat(0.1, 0.3), 'spacingStaff': rfloat(1, 12)})

            self.tk.getPageCount()
            svg = self.tk.renderToSVG()
            svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")
            
            if check_generated_systems == True:
                generated_systems = self.count_class_occurrences(svg_file=svg, class_name='grpSym')
            else:
                generated_systems = n_sys_generate
        
        pngfile = svg2png(bytestring=svg, background_color='white')
        pngfile = cv2.imdecode(np.frombuffer(pngfile, np.uint8), -1)
        
        if cut_height == True:
            height = self.find_image_cut(pngfile)
            pngfile = pngfile[:height + padding, :]

        x = pngfile
        
        if add_texture == True:
            texture = Image.open(random.choice(self.textures))
            img_width, img_height = x.shape[1], x.shape[0]
            #Check if the texture is smaller than the image. If so, resize it to the image size
            if texture.size[0] < img_width or texture.size[1] < img_height:
                texture = texture.resize((img_width, img_height))
            
            x = self.inkify_image(x)
            x = np.array(x)
            music_image = Image.fromarray(x)
            left = random.randint(0, texture.size[0] - img_width)
            top = random.randint(0, texture.size[1] - img_height)
            texture = texture.crop((left, top, left + img_width, top + img_height))
            inverted_music_image = ImageOps.invert(music_image.convert('RGB'))
            mask = inverted_music_image.convert("L")
            texture.paste(music_image, mask=mask)
            x = texture#cv2.cvtColor(np.array(texture), cv2.COLOR_RGB2BGR)
        else:
            x = np.array(x)
        
        x = cv2.cvtColor(np.array(x), cv2.COLOR_BGR2RGB)
        
        width = int(np.ceil(pngfile.shape[1] * reduce_ratio))
        height = int(np.ceil(pngfile.shape[0] * reduce_ratio))
        x = cv2.resize(x, (width, height))

        if self.tokenization_method == "ekern":
            sequence = "".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").split(" ")
        
        if self.tokenization_method == "bekern":
            sequence = "".join(sequence).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")
            

        return x, ['<bos>'] + sequence[4:-1] + ['<eos>']