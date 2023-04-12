# -*- coding: utf-8 -*-
'''
Modified and edited by Jagabandhu Mishra, 27/07/2022
This script is used to compute the cosine similarity

@code
Re-written by: 
    Mrinmoy Bhattacharjee, 
    Senior Project Engineer, 
    Dept. of EE, IIT Dharwad. 
    March 10, 2023
    
    The audio is not splitted into chunks now. The code now generates 
    annonated .textgrid files with the speech and sil labels that can 
    be directly loaded into the Praat software to update the labels

'''

import glob
import librosa
from speechbrain.pretrained import EncoderClassifier
import numpy as np
import torch
from scipy import linalg
import csv
import os
from scipy.io import wavfile


# wav_dir = '../VAD_Demo_fold/'
# threshold = 0.6

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wav_dir', type=str)
parser.add_argument('--vad_dir', type=str)
parser.add_argument('--threshold', type=float)
args = parser.parse_args()

wav_dir = args.wav_dir
vad_dir = args.vad_dir
threshold = args.threshold


print("Load wav from " + str(wav_dir))
list_folder = str(vad_dir) + "/*.csv"
list_folder = glob.glob(list_folder)
print(list_folder)


classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb')

#define cospair function

def cos_pair(a,b):
    return np.dot(a,b.T)/linalg.norm(a)/linalg.norm(b)


for threshold in [0.2, 0.4, 0.6, 0.8]:
    opDir = vad_dir + '/' + str(threshold) + '/'
        
    if not os.path.exists(opDir):
        os.makedirs(opDir)
    wav_opDir = opDir + '/wav/'
    if not os.path.exists(wav_opDir):
        os.makedirs(wav_opDir)
    
    #min_mat save min cosine pair of 1 wav; min_path save path of wav
    for csv_file in list_folder:
        min_mat = []
        min_path = []
        start_mat = []
        end_mat = []
        dur_mat = []
        label_mat = []
        
        fName = csv_file.split('/')[-1].split('.')[0]
        
        wav_path = wav_dir + '/' + fName + '.wav'
        print(f'wav_path={wav_path} csv_file={csv_file}')
        Xin, frequency = librosa.load(wav_path, mono=True, sr=None)
    
        cosine_csv_file = opDir + '/' +  csv_file.split('/')[-1].replace('.csv', '_thresholded.csv')
        with open(cosine_csv_file, 'w+') as csv_fid:
            csv_fid.write('start,end,label,duration,label_change\n')
    
    
        # cosine_textgrid_file = csv_file.replace('.csv', '_thresholded.textgrid')
        # with open(cosine_textgrid_file, 'w+') as textgrid_fid:
        #     textgrid_fid.write('    File type = "ooTextFile"\n')
        #     textgrid_fid.write('    Object class = "TextGrid"\n')
        #     textgrid_fid.write('    xmin = 0\n')
        #     textgrid_fid.write(f'    xmax = {len(Xin)/frequency}\n')
        #     textgrid_fid.write('    tiers? <exists>\n')
        #     textgrid_fid.write('    size = 1\n')
        #     textgrid_fid.write('    item []:\n')
    
        num_rows = 0
        with open(csv_file, 'r') as fid:
            lines = fid.readlines()
            num_rows = len(lines)-1
    
        # with open(cosine_textgrid_file, 'a+') as textgrid_fid:
        #     textgrid_fid.write('       item [1]:\n')
        #     textgrid_fid.write('          class = "IntervalTier"\n')
        #     textgrid_fid.write('          name = "speech"\n')
        #     textgrid_fid.write("          xmin = 0\n")
        #     textgrid_fid.write(f"          xmax = {len(Xin)/frequency}\n")
        #     textgrid_fid.write(f"          intervals: size = {num_rows-1}\n")
        
        segment_count = 0
        with open(csv_file, 'r') as fid:
            readlines = csv.DictReader(fid)
            row_count = 1
            for row in readlines:
                smpStart = int(row['start'])
                smpEnd = int(row['end'])
                duration = float(row['duration'])
                label = row['label']
                signal = Xin[smpStart:smpEnd]
                print(f"{csv_file.split('/')[-1]} ({row_count}/{num_rows}) signal={len(signal)} start={smpStart} end={smpEnd} dur={duration}")
                row_count += 1
                if label=='sil':
                    # with open(cosine_textgrid_file, 'a+') as textgrid_fid:
                    #     textgrid_fid.write(f'          intervals [{row_count}]:\n')
                    #     textgrid_fid.write(f"                xmin = {smpStart/frequency}\n")
                    #     textgrid_fid.write(f"                xmax = {smpEnd/frequency}\n")
                    #     textgrid_fid.write("                text = \"sil\"\n")
    
                    continue
                            
                # frequency, signal = wavfile.read(x)
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
            
                matrix_audio = [ [0]*(len(audio)) for i in range(len(audio))]
                for i in range(len(audio)):
                    for j in range(len(audio)):
                        matrix_audio[i][j]=(cos_pair(audio[i], audio[j]))
                        # print(matrix_audio)
                
                matrix_audio_min_list = []
                for r in matrix_audio:
                    if len(r)==0:
                        continue
                    matrix_audio_min_list.append(min(r))
                if len(matrix_audio_min_list)==0:
                    mymin = 0
                else:
                    mymin = min(matrix_audio_min_list)
                
                if mymin>threshold:
                    thresholded_label = 'speech'
                    with open(cosine_csv_file, 'a+') as csv_fid:
                        csv_fid.write(f'{smpStart/frequency},{smpEnd/frequency},{thresholded_label},{np.round((smpEnd-smpStart)/frequency,2)},No\n')
                else:
                    thresholded_label = 'sil'
                    with open(cosine_csv_file, 'a+') as csv_fid:
                        csv_fid.write(f'{smpStart/frequency},{smpEnd/frequency},{thresholded_label},{np.round((smpEnd-smpStart)/frequency,2)},Yes\n')
                    chunk_fName = wav_opDir + '/' + fName + '_' + str(segment_count) + '.wav'
                    wavfile.write(chunk_fName, frequency, signal)
                    segment_count += 1
                # with open(cosine_textgrid_file, 'a+') as textgrid_fid:
                #     textgrid_fid.write(f'          intervals [{row_count}]:\n')
                #     textgrid_fid.write(f"                xmin = {smpStart/frequency}\n")
                #     textgrid_fid.write(f"                xmax = {smpEnd/frequency}\n")
                #     textgrid_fid.write(f"                text = \"{thresholded_label}\"\n")
