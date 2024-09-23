import os
import re
import numpy as np
from eval_functions import compute_poliphony_metrics

def locate_token(line, token):
    dynam_positions = []
    for idx, token in enumerate(line):
        if "dynam" in token:
            dynam_positions.append(idx)
    return dynam_positions

def erase_numbers_in_tokens_with_equal(tokens):
    return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

def filter_by_content(line, patterns):
    for pattern in patterns:
        if pattern in line:
            return False
    return True

def simplify_tokens(tokens):
    simplified_tokens = []
    for token in tokens:
        token = re.sub(r'::+', '', token)
        simplified_tokens.append(token)
    return simplified_tokens

def clean_kern(krn_path):
    true_lines = []
    with open(krn_path, 'r') as f:
        lines = f.readlines()
    
    lines = [line.replace('\n', '').replace('·', '').replace('@', '') for line in lines if filter_by_content(line, ['*part', '*I', '*staff', '!'])]
    
    lines = [erase_numbers_in_tokens_with_equal([line])[0].split('\t') for line in lines]
    
    dynam_location = locate_token(lines[0], token="dynam")
    krn_filtered_lines = []
    offset = 0
    for line in lines[1:]:
        reconstructed_line = []
        for idx, token in enumerate(line):
            if idx not in list(np.array(dynam_location) + offset):
                reconstructed_line.append(token)
        
        rec_line = "\t".join(reconstructed_line)
        rec_line = rec_line.replace("/", "")
        rec_line = rec_line.replace("\\", "")
        rec_line = re.sub(r'J+', 'J', rec_line)
        rec_line = re.sub(r'L+', 'L', rec_line)
        
        krn_filtered_lines.append(rec_line)
        
        splitter_location = locate_token(line, token="*^")
        splitter_location = [loc for loc in splitter_location if loc < dynam_location]
        offset += len(splitter_location)
        
        joiner_location = locate_token(line, token="*v")
        joiner_location = [loc for loc in joiner_location if loc < dynam_location]
        offset += len(splitter_location)//2
    
    return "\n".join(krn_filtered_lines)



