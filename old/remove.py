'''
Modified and edited by Jagabandhu Mishra, 27/07/2022
This script is used to remove the files lesser than the threshold (cosine similarity)

'''

import argparse
from ast import arg
import glob
import librosa
import torchaudio
import numpy as np
from scipy.io import wavfile
import numpy as np
from scipy.io.wavfile import write
import torch
import os
from scipy import linalg, mat, dot
import pandas as pd
import seaborn as sns
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_csv', type=str,
                        help='file_csv')
parser.add_argument('-t', '--thresh_hold', type=str,
                        help='thresh_hold')
args = parser.parse_args()
#%%
# args.file_csv='data/mlsvyou_hindi_data/CCS_demo_CSV/demo.csv'
# args.thresh_hold=0.2


#%%
thresh_hold = args.thresh_hold

file_csv = pd.read_csv(args.file_csv)
for i in range(len(file_csv)):
    if ((file_csv['MinCos'][i])<float(thresh_hold)):
        os.remove(file_csv['Path'][i])