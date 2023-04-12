'''
Modified and edited by Jagabandhu Mishra, 27/07/2022
This script is used to split the youtube file into VAD split segments 

'''


SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)
from IPython.display import Audio
from pprint import pprint
import glob
import os
import librosa
import argparse


#%%
 
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils



#%%
parser = argparse.ArgumentParser()
parser.add_argument('--folder_file_wav', type=str)
parser.add_argument('--save_dir', type=str)

args = parser.parse_args()
#%% only for checking

# args.folder_file_wav='data/mlsvyou_hindi_data/Demo_fold'
# args.save_dir='data/mlsvyou_hindi_data/VAD_Demo_fold'



#%%

def vad():
    for name in glob.glob(args.folder_file_wav + "/*"):
        j = 0
        wav = read_audio(name, sampling_rate = SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=SAMPLING_RATE)

        sum = 0
        k = 0   
        speech_timestamps_mini = []
        mini_audio = []

        while(True):
            sum =  sum + speech_timestamps[k]['end'] - speech_timestamps[k]['start']
            speech_timestamps_mini.append(speech_timestamps[k])
                
            if k < len(speech_timestamps)-1:
                k = k + 1
                if sum >= 48000:      
                    mini_audio.append(collect_chunks(speech_timestamps_mini, wav))
                    speech_timestamps_mini.clear()
                    sum = 0
                    continue
                else:
                    continue
            else:
                mini_audio.append(collect_chunks(speech_timestamps_mini, wav))
                speech_timestamps_mini.clear()
                sum = 0
                break   
        
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if not os.path.exists(args.save_dir + '/' + os.path.splitext(os.path.basename(name))[0].replace(" ", "_")):
            os.mkdir(args.save_dir + '/'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_"))
        for i in mini_audio:
            save_audio(args.save_dir + '/'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '/' + str(j) + '.wav', i, sampling_rate = SAMPLING_RATE)
                
            j = j + 1 

vad()
            

#%%

def re_vad():   
    for name in glob.glob(args.save_dir + '/*' ):
        j = 30000
        for name3 in glob.glob(name + '/*'):
            file1 = librosa.get_duration(filename = name3)
            if file1 >= 10.0:
                wav = read_audio(name3, sampling_rate = SAMPLING_RATE)
                speech_timestamps1 = get_speech_timestamps(wav, model, threshold=0.9, sampling_rate=SAMPLING_RATE)
                sum = 0
                k = 0   
                speech_timestamps_mini = []
                mini_audio = []

                if len(speech_timestamps1) != 0:
                    while(True):
                        sum =  sum + speech_timestamps1[k]['end'] - speech_timestamps1[k]['start']
                        
                        speech_timestamps_mini.append(speech_timestamps1[k])
                        
                        if k < len(speech_timestamps1)-1:
                            k = k + 1
                            if sum >= 48000:      
                                mini_audio.append(collect_chunks(speech_timestamps_mini, wav))
                                speech_timestamps_mini.clear()
                                sum = 0
                                continue
                            else:
                                continue
                        else:
                            mini_audio.append(collect_chunks(speech_timestamps_mini, wav))
                            speech_timestamps_mini.clear()
                            sum = 0
                            break
                    else:
                        continue   
                            
                for i in mini_audio:
                    print(i)
                    save_audio(args.save_dir + '/'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '/' + str(j) + '.wav', i, sampling_rate = SAMPLING_RATE)
                    j = j + 1  


#%%

def remove():
    for i in glob.glob(args.save_dir + '/*' ):
        k = 0
        for j in glob.glob(i + '/*'):
            if librosa.get_duration(filename = j) < 3 or librosa.get_duration(filename = j) > 10:
                os.remove(j)
            else:
                os.rename(j, i + '/audio_' + str(k) + '.wav')
                k= k+1

remove()
        

