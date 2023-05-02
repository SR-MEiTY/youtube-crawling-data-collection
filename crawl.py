# -*- coding: utf-8 -*-
'''
@code
Modified by: 
    Mrinmoy Bhattacharjee, 
    Senior Project Engineer, 
    Dept. of EE, IIT Dharwad. 
    March 10, 2023
    
'''

import os
import argparse
import librosa
import soundfile as sf
import re
import pytube
from pytube import YouTube

from pytube import innertube
innertube._cache_dir = os.path.join('~/.cache/', 'pytube_cache')
if not os.path.exists(innertube._cache_dir):
    os.makedirs(innertube._cache_dir)
innertube._token_file = os.path.join(innertube._cache_dir, 'tokens.json')

import torchaudio
import torchaudio.functional as F
from moviepy.editor import *
import glob
from pydub import AudioSegment
#import  shutil as sh
import csv
import sys

parser = argparse.ArgumentParser()
#download setting
parser.add_argument('--url_playlist', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--vad_dir', type=str)
##parser.add_argument('--save_video', type=str)

args = parser.parse_args()
url_playlist = args.url_playlist
save_dir = args.save_dir
vad_dir = args.vad_dir

# video index to continue crawling, index = 0 means the first video
START_INDEX = 0

try:
    playlist_title = pytube.Playlist(url_playlist).title
    save_dir = save_dir + '/' + playlist_title.replace(' ', '_') + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vad_dir = vad_dir + '/' + playlist_title.replace(' ', '_') + '/'
    if not os.path.exists(vad_dir):
        os.makedirs(vad_dir)
    print('Number of videos in playlist: %s' % len(pytube.Playlist(url_playlist).video_urls))
except:
    print('Network error')
    sys.exit(0)


def download_video():

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    playlist = pytube.Playlist(url_playlist)

    number = 0
    video = playlist.video_urls
    for i in range(0, len(video)):
        number = number + 1
        #if number == 2:
        #    break
        id = re.match('^[^v]+v=(.{11}).*', video[i])
            
        ''' Added by Mrinmoy Bhattacharjee, March 16, 2023 '''
        fName = ''
        if os.path.exists('session_id.csv'):
            with open('session_id.csv', 'r') as fid:
                reader = csv.DictReader(fid)
                for row in reader:
                    if (row['youtube_id']==id.group(1)):
                        fName = row['session_id']

        if fName=='':
            # print('Requested playlist video not found in the session_id.csv file. Skipping')
            fName = video[i].split('=')[1]
        
        if os.path.exists(save_dir + '/' + fName+'.mp4'):
            print('Requested video already downloaded')
        else:            
            try:
                yt = YouTube(
                    video[i], 
                    use_oauth=True, 
                    allow_oauth_cache=True
                    )
                print(f'Downloading #{number} Filename={fName} Title={yt.title}')
                yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                yt.download(save_dir, filename=fName + '.mp4')
                print(f'\t{fName} downloaded.')
            except:
                print('\tDownload Error')
            
        ''' ----------------------------------- '''

        #sh.copyfile(save_dir+'/'+id.group(1)+'.mp4',save_video+'/'+id.group(1)+'.mp4')
        #print(number)

def convert_to_mp3():
    for i in glob.glob(save_dir + '/*.mp4'):
        mp3_fName = os.path.splitext(i)[0] + '.mp3'
        if os.path.exists(mp3_fName):
            continue
        videoclip = VideoFileClip(i)
        audioclip = videoclip.audio
        audioclip.write_audiofile(os.path.splitext(i)[0] + '.mp3')

def convert_to_wav():
    for i in glob.glob(save_dir + '/*.mp3'):
        wav_fName = vad_dir + '/' + os.path.splitext(i)[0].split('/')[-1] + '.wav'
        if os.path.exists(wav_fName):
            continue
        sound = AudioSegment.from_mp3(i)
        sound.export(vad_dir + '/' + os.path.splitext(i)[0].split('/')[-1] + '.wav', format="wav")

def resample_wav():
    for i in glob.glob(save_dir + '/*.wav'):
        print(i)
        try:
            y, sr = torchaudio.load(i)       
            y_16k = F.resample(y, sr, 16000)
            y_16k = y_16k.numpy()
            y_mono = librosa.to_mono(y_16k)
            sf.write(i, y_mono, 16000)
        except:
            bruh = 0

def get_resample():
    k = 0
    for i in glob.glob(save_dir + '/*.wav'):
        print(librosa.get_samplerate(i))
        k = k+1
    print(k)    

def remove():
    fileMp4 = glob.glob(save_dir + '/*.mp4')
    fileMp3 = glob.glob(save_dir + '/*.mp3')

    print("Number of files_mp4: ", len(fileMp4))
    print("Number of files_mp3: ", len(fileMp3))
        
    for file1 in fileMp4:
        os.remove(file1)
    for file2 in fileMp3:
        os.remove(file2)

download_video()
convert_to_mp3()
convert_to_wav()
# resample_wav()
# get_resample()
# remove()
