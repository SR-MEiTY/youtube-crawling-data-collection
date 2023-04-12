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
from pytube import YouTube
import pytube
import torchaudio
import torchaudio.functional as F
from moviepy.editor import *
import glob
from pydub import AudioSegment
#import  shutil as sh
import csv

parser = argparse.ArgumentParser()
#download setting
parser.add_argument('--url_playlist', type=str)
parser.add_argument('--save_dir', type=str)
##parser.add_argument('--save_video', type=str)

args = parser.parse_args()

# video index to continue crawling, index = 0 means the first video
START_INDEX = 0

def download_video():
    playlist = pytube.Playlist(args.url_playlist)
    print('Number of videos in playlist: %s' % len(playlist.video_urls))
    number = 0
    video = playlist.video_urls
    for i in range(0, len(video)):
        number = number + 1
        #if number == 2:
        #    break
        id = re.match('^[^v]+v=(.{11}).*', video[i])
        yt = YouTube(video[i])
        yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        # yt.download(args.save_dir, filename=id.group(1) + '.mp4')
            
        ''' Added by Mrinmoy Bhattacharjee, March 16, 2023 '''
        fName = ''
        with open('../../session_id.csv', 'r') as fid:
            reader = csv.DictReader(fid)
            for row in reader:
                if (row['youtube_id']==id.group(1)):
                    fName = row['session_id']
                    break
        if fName=='':
            print('Requested playlist video not found in the session_id.csv file. Skipping')
        elif os.path.exists(fName+'.mp4'):
            print('Requested video already downloaded')
        else:
            yt.download(args.save_dir, filename=fName + '.mp4')
        ''' ----------------------------------- '''

        #sh.copyfile(args.save_dir+'/'+id.group(1)+'.mp4',args.save_video+'/'+id.group(1)+'.mp4')
        print(number)

def convert_to_mp3():
    for i in glob.glob(args.save_dir + '/*.mp4'):
        videoclip = VideoFileClip(i)
        audioclip = videoclip.audio
        audioclip.write_audiofile(os.path.splitext(i)[0] + '.mp3')

def convert_to_wav():
    for i in glob.glob(args.save_dir + '/*.mp3'):
        sound = AudioSegment.from_mp3(i)
        sound.export(os.path.splitext(i)[0] + '.wav', format="wav")

def resample_wav():
    for i in glob.glob(args.save_dir + '/*.wav'):
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
    for i in glob.glob(args.save_dir + '/*.wav'):
        print(librosa.get_samplerate(i))
        k = k+1
    print(k)    

def remove():
    fileMp4 = glob.glob(args.save_dir + '/*.mp4')
    fileMp3 = glob.glob(args.save_dir + '/*.mp3')

    print("Number of files_mp4: ", len(fileMp4))
    print("Number of files_mp3: ", len(fileMp3))
        
    for file1 in fileMp4:
        os.remove(file1)
    for file2 in fileMp3:
        os.remove(file2)

download_video()
# convert_to_mp3()
# convert_to_wav()
# resample_wav()
# get_resample()
# remove()
