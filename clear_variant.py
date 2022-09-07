##### Zhores experiments

### 1) Preprations to mGENRE launch on Zhores


# built in imports
import subprocess
import sys
import os

# imports
import warnings
warnings.filterwarnings("ignore")


# function to install packages from .py file
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# install packages from .py
install("gitpython")
from git.repo.base import Repo





# clone git repo
os.system("rm -rf mGENRE_MEL")
Repo.clone_from("https://github.com/SergeyPetrakov/mGENRE_MEL", "mGENRE_MEL")

# set mGENRE_MEL as working
os.chdir(os.getcwd() + "/mGENRE_MEL")
init_dir = os.getcwd()
os.system("pip uninstall -y numpy")
os.system("pip install numpy")


print(init_dir)

# KILT
install("tqdm")
install("spacy>=2.1.8")

KILTdir = init_dir + "/KILT"
os.system("rm -rf KILTdir")
Repo.clone_from("https://github.com/facebookresearch/KILT.git", "KILT")
os.chdir(init_dir + "/KILT")
os.system("python setup.py install")
sys.path.append(os.getcwd())
os.chdir(init_dir)

print("#"*100)
print("KILT installation is finished")
print("#"*100)

# GENRE
GENREdir = init_dir + "/GENRE"
os.system("rm -rf GENREdir")
Repo.clone_from("https://github.com/SergeyPetrakov/GENRE", "GENRE")
os.chdir(init_dir + "/GENRE")
os.system("pip install ./")
os.system("python ./setup.py build develop install")
os.system("pip install sentencepiece marisa_trie")
sys.path.append(os.getcwd())
os.chdir(init_dir)

print("#"*100)
print("GENRE installation is finished")
print("#"*100)

# Fairseq
Fairseqdir = init_dir + "/fairseq"
os.system("rm -rf Fairseqdir")
Repo.clone_from("https://github.com/SergeyPetrakov/fairseq", "fairseq", branch="fixing_prefix_allowed_tokens_fn")
os.chdir(Fairseqdir)
os.system("sed -i -e '26,27d' fairseq/registry.py")
os.system("pip install --editable ./")
os.system("python setup.py build develop")
os.system("python setup.py install")
sys.path.append(os.getcwd())
os.chdir(init_dir)

print("#"*100)
print("Fairseq installation is finished")
print("#"*100)

# additionally

os.chdir(Fairseqdir)
os.system("pip install --editable ./")
os.chdir(init_dir)
os.system("pip uninstall -y numpy")
install("numpy")
install("gdown")
install("stanza")

# import libraries
import gdown
import json
import pickle
import torch
import gc
import re
install("wikidata")
from wikidata.client import Client
install("pandas")
# install("random")
# install("scipy")
# install("string")
# install("json")
# install("itertools")
# install("datetime")
install("matplotlib")

import pandas as pd
import numpy as np
import random
import scipy.stats
import string
import json
from itertools import compress
from datetime import datetime
import matplotlib.pyplot as plt
# import stanza
from tqdm import tqdm

print("All libraries are imported")

print("Upload data and model")
# model
os.system("rm -rf fairseq_multilingual_entity_disambiguation*")
os.system("wget https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz")
os.system("tar -xvf fairseq_multilingual_entity_disambiguation.tar.gz")
# data
os.system("wget https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl")
os.system("wget http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl")


os.chdir(init_dir)
print(init_dir)

#Quick Start. Multilingual Autoregressive Entity Linking. GENRE

path_1 = init_dir+'/GENRE'

#sys.path.insert(0, path_1)
from genre.trie import Trie
from genre.trie import MarisaTrie

print("Trie, MarisaTrie and mGENRE are imported")

#sys.path.insert(0, init_dir)
with open("lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
    lang_title2wikidataID = pickle.load(f)

# with open("titles_lang_all105_trie_with_redirect.pkl", "rb") as f:
#     trie = Trie.load_from_dict(pickle.load(f))

with open("titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)

print("lang_title2wikidataID and trie done")

from genre.fairseq_model import mGENRE

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
# generate Wikipedia titles and language IDs
model_mGENRE = mGENRE.from_pretrained("fairseq_multilingual_entity_disambiguation").eval()
model_mGENRE.to(device)


sentences = ["[START] The founder of the theory of relativity [END] received the Nobel Prize."]
print(model_mGENRE.sample(
    sentences,
    beam = 5,
    prefix_allowed_tokens_fn=lambda batch_id, sent: [
        e for e in trie.get(sent.tolist())
        if e < len(model_mGENRE.task.target_dictionary)
    ],
    text_to_id=lambda x: max(lang_title2wikidataID[tuple(reversed(x.split(" >> ")))], key=lambda y: int(y[1:])),
    marginalize=True,
    verbose = True
))



