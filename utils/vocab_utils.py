import os

import numpy as np
from os import path
from loguru import logger


@logger.catch
def check_and_retrieveVocabulary(YSequences, pathOfSequences, nameOfVoc, save=True):
    w2ipath = pathOfSequences + "/" + nameOfVoc + "w2i.npy"
    i2wpath = pathOfSequences + "/" + nameOfVoc + "i2w.npy"

    w2i = []
    i2w = []

    if not path.isdir(pathOfSequences):
        os.mkdir(pathOfSequences)

    if path.isfile(w2ipath):
        w2i = np.load(w2ipath, allow_pickle=True).item()
        i2w = np.load(i2wpath, allow_pickle=True).item()
    else:
        w2i, i2w = make_vocabulary(YSequences, pathOfSequences, nameOfVoc, save)

    return w2i, i2w

def make_vocabulary(YSequences, pathToSave, nameOfVoc, save=True):
    vocabulary = set()
    for samples in YSequences:
        for element in samples:
                vocabulary.update(element)

    #Vocabulary created
    w2i = {symbol:idx+1 for idx,symbol in enumerate(vocabulary)}
    i2w = {idx+1:symbol for idx,symbol in enumerate(vocabulary)}
    
    w2i['<pad>'] = 0
    i2w[0] = '<pad>'

    #Save the vocabulary
    if save:
        np.save(pathToSave + "/" + nameOfVoc + "w2i.npy", w2i)
        np.save(pathToSave + "/" + nameOfVoc + "i2w.npy", i2w)

    return w2i, i2w