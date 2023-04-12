# -*- coding: utf-8 -*-
'''
Modified and edited by Jagabandhu Mishra, 27/07/2022
This script is used to compute the cosine similarity

'''

import argparse
import glob
import librosa
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from scipy.io import wavfile
import numpy as np
from scipy.io.wavfile import write
import torch
import os
import numpy as np
from scipy import linalg, mat, dot
import pandas as pd
import seaborn as sns
from pathlib import Path
#%%
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--wav_dir', type=str,
                        help='wav_dir')
parser.add_argument('-f', '--file_csv', type=str,
                        help='file_csv')
args = parser.parse_args()

#%%
# args.wav_dir='data/mlsvyou_hindi_data/VAD_Demo_fold'
# args.file_csv='data/mlsvyou_hindi_data/CCS_demo_CSV/demo.csv'

#%%

print("Load wav from " + str(args.wav_dir))
list_folder = str(args.wav_dir) + "/*/"
print(list_folder)
list_folder = glob.glob(list_folder)
#%%
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
#%%
#define cospair function

def cos_pair(a,b):
  return np.dot(a,b.T)/linalg.norm(a)/linalg.norm(b)

#min_mat save min cosine pair of 1 wav; min_path save path of wav
min_mat = []
min_path = []
for le in range(len(list_folder)):
    x = glob.glob(str(list_folder[le]) +  '*.wav')
    
    for _ in range(len(x)):
        frequency, signal = wavfile.read(x[_])
        slice_length = 1.2 # in seconds
        overlap = 0.2 # in seconds
        slices = np.arange(0, len(signal)/frequency, slice_length-overlap, dtype=int)
        i = 0
        audio = []
        matrix_audio = []
        for start, end in zip(slices[:-1], slices[1:]):
            i = i + 1
            start_audio = start * frequency
            end_audio = (end + overlap)* frequency
            audio_slice = signal[int(start_audio): int(end_audio)]
            audio_slice = audio_slice.reshape(1,-1)
            audio_slice = torch.tensor(audio_slice)
            audio_slice = classifier.encode_batch(audio_slice)
            audio_slice = audio_slice.squeeze()
            audio.append(audio_slice)
        try:
            matrix_audio = [ [0]*(len(audio)) for i in range(len(audio))]
            for i in range(len(audio)):
                for j in range(len(audio)):
                    matrix_audio[i][j]=(cos_pair(audio[i], audio[j]))
                    # print(matrix_audio)
            mymin = min([min(r) for r in matrix_audio])
            min_mat.append(mymin)
            min_path.append(x[_])
        except:
            print(x[_]) 
            os.remove(x[_])
    print(le)

#%%
#   import numpy as np
#   import matplotlib.pyplot as plt


#   fig = plt.figure(figsize=(6, 3.2))

#   ax = fig.add_subplot(111)
#   ax.set_title('color')
#   plt.imshow(matrix_audio)
#   ax.set_aspect('equal')

#   cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
#   cax.get_xaxis().set_visible(False)
#   cax.get_yaxis().set_visible(False)
#   cax.patch.set_alpha(0)
#   cax.set_frame_on(False)
#   plt.colorbar(orientation='vertical')
#   plt.show()

#%%

data = []
for i in range(len(min_mat)):
    data.append([min_mat[i],min_path[i]])
print(data)


data = pd.DataFrame([min_mat,min_path]) #Each list would be added as a row
data = data.transpose() #To Transpose and make each rows as columns
data.columns=['MinCos','Path'] #Rename the columns
print("Save csv to " + str(args.file_csv))
my_file = Path(args.file_csv)
try:
  data.to_csv(args.file_csv)
except:
  print("No dir name " + str(args.file_csv))

  

