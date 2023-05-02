'''
Modified and edited by Jagabandhu Mishra, 27/07/2022
This script is used to split the youtube file into VAD split segments

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


SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)
import glob
import os
import librosa
import csv
from scipy.io import wavfile
import pytube
from pytube import YouTube
import sys


model, utils = torch.hub.load(repo_or_dir='./silero-vad-master/', model='silero_vad', source='local', force_reload=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--url_playlist', type=str)
parser.add_argument('--video_folder', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()
video_folder = args.video_folder
save_dir = args.save_dir


try:
    playlist = pytube.Playlist(args.url_playlist)
    playlist_title = playlist.title
    video_folder = video_folder + '/' + playlist_title.replace(' ', '_') + '/'
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    save_dir = save_dir + '/' + playlist_title.replace(' ', '_') + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Number of videos in playlist: %s' % len(playlist.video_urls))
except:
    print('Network error')
    sys.exit(0)



def vad_new():
    for name in glob.glob(save_dir + "/*.wav"):
        print(f'name={name}')
        # wav_file = save_dir + '/'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '.wav'
        vad_file = save_dir + '/'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '.csv'
        
        if not os.path.exists(vad_file):
            j = 0
            wav = read_audio(name, sampling_rate = SAMPLING_RATE)
            # print(wav.cpu().detach().numpy())
            
            # if not os.path.exists(wav_file):
            #     wavfile.write(wav_file, SAMPLING_RATE, wav.cpu().detach().numpy())
            speech_timestamps = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=SAMPLING_RATE)
    
            sum = 0
            k = 0
            speech_timestamps_mini = []
            mini_audio = []
    
            with open(vad_file, 'w+') as fid:
                fid.write('start,end,label,duration\n')
                        
            prev_end = 0
            while(True):
                sum =  sum + speech_timestamps[k]['end'] - speech_timestamps[k]['start']
                # print(f'timestamps={speech_timestamps[k]}')
                speech_timestamps_mini.append(speech_timestamps[k])
                    
                if k < len(speech_timestamps)-1:
                    k = k + 1
                    if sum >= 48000:      
                        mini_audio.append(collect_chunks(speech_timestamps_mini, wav))
                        # print(f'speech_timestamps_mini={speech_timestamps_mini}')
                        for chunks in speech_timestamps_mini:
                            with open(vad_file, 'a+') as fid:
                                if prev_end==0:
                                    if not int(chunks['start'])==0:
                                        chunk_dur = round(chunks['start']/SAMPLING_RATE,2)
                                        fid.write(f"0,{chunks['start']},others,{chunk_dur}\n")
    
                                    prev_end = int(chunks['end'])
                                    chunk_dur = round((chunks['end']-chunks['start'])/SAMPLING_RATE,2)
                                    fid.write(f"{chunks['start']},{chunks['end']},single_speaker,{chunk_dur}\n")
                                else:
                                    chunk_dur = round((chunks['start']-prev_end)/SAMPLING_RATE,2)
                                    fid.write(f"{prev_end},{chunks['start']},others,{chunk_dur}\n")
                                    prev_end = int(chunks['end'])
                                    chunk_dur = round((chunks['end']-chunks['start'])/SAMPLING_RATE, 2)
                                    fid.write(f"{chunks['start']},{chunks['end']},single_speaker,{chunk_dur}\n")
                        speech_timestamps_mini.clear()
                        sum = 0
                        continue
                    else:
                        continue
                else:
                    mini_audio.append(collect_chunks(speech_timestamps_mini, wav))
                    # print(f'speech_timestamps_mini={speech_timestamps_mini}')
                    for chunks in speech_timestamps_mini:
                        with open(vad_file, 'a+') as fid:
                            if prev_end==0:
                                if not int(chunks['start'])==0:
                                    chunk_dur = round(chunks['start']/SAMPLING_RATE,2)
                                    fid.write(f"0,{chunks['start']},others,{chunk_dur}\n")
    
                                prev_end = int(chunks['end'])
                                chunk_dur = round((chunks['end']-chunks['start'])/SAMPLING_RATE,2)
                                fid.write(f"{chunks['start']},{chunks['end']},single_speaker,{chunk_dur}\n")
                            else:
                                chunk_dur = round((chunks['start']-prev_end)/SAMPLING_RATE,2)
                                fid.write(f"{prev_end},{chunks['start']},others,{chunk_dur}\n")
                                prev_end = int(chunks['end'])
                                chunk_dur = round((chunks['end']-chunks['start'])/SAMPLING_RATE, 2)
                                fid.write(f"{chunks['start']},{chunks['end']},single_speaker,{chunk_dur}\n")
                    speech_timestamps_mini.clear()
                    sum = 0
                    break
                    
                j = j + 1 

        vad_textgrid_file = save_dir + '/'  + os.path.splitext(os.path.basename(name.split('.')[0]))[0].replace(" ", "_") + '.textgrid'
        if os.path.exists(vad_textgrid_file):
            print(f'{name} TextGrid already exists')
            continue

        with open(vad_textgrid_file, 'w+') as textgrid_fid:
            textgrid_fid.write('    File type = "ooTextFile"\n')
            textgrid_fid.write('    Object class = "TextGrid"\n')
            textgrid_fid.write('    xmin = 0\n')
            textgrid_fid.write(f'    xmax = {len(wav)/SAMPLING_RATE}\n')
            textgrid_fid.write('    tiers? <exists>\n')
            textgrid_fid.write('    size = 1\n')
            textgrid_fid.write('    item []:\n')
        
        num_lines = 0
        with open(vad_file, 'r') as vad_fid:
            lines = vad_fid.readlines()
            num_lines = len(lines)

        with open(vad_file, 'r') as vad_fid:
            readlines = csv.DictReader(vad_fid)
            with open(vad_textgrid_file, 'a+') as textgrid_fid:
                textgrid_fid.write('       item [1]:\n')
                textgrid_fid.write('          class = "IntervalTier"\n')
                textgrid_fid.write('          name = "speaker_tier"\n')
                textgrid_fid.write("          xmin = 0\n")
                textgrid_fid.write(f"          xmax = {len(wav)/SAMPLING_RATE}\n")
                textgrid_fid.write(f"          intervals: size = {num_lines-1}\n")
            row_count = 1
            for row in readlines:
                with open(vad_textgrid_file, 'a+') as textgrid_fid:
                    textgrid_fid.write(f'          intervals [{row_count}]:\n')
                    textgrid_fid.write(f"                xmin = {float(row['start'])/SAMPLING_RATE}\n")
                    textgrid_fid.write(f"                xmax = {float(row['end'])/SAMPLING_RATE}\n")
                    textgrid_fid.write(f"                text = \"{row['label']}\"\n")
                row_count += 1


vad_new()
            


def re_vad_new():   
    for name in glob.glob(save_dir + '/*.csv' ):
        re_vad_file = save_dir + '/.'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '.csv'
        print(f're_vad_file={re_vad_file} name={name}')
        
        # print(f'name={name}')
        media_file = glob.glob(name.split('.csv')[0]+'*.wav')
        # print(f'media_file={media_file}')
        if len(media_file)==0:
            continue
        wav = read_audio(media_file[0], sampling_rate = SAMPLING_RATE)

        with open(re_vad_file, 'w+') as fid:
            fid.write('start,end,label,duration\n')

        with open(name, 'r') as fid:
            read_lines = csv.DictReader(fid)
            all_chunks = 0
            for row in read_lines:
                # print(f'row={row}')
                smpStart = int(float(row['start'])*len(wav))
                smpEnd = int(float(row['end'])*len(wav))
                file_dur = float(row['duration'])
                
                if (file_dur >= 10.0) and (row['label']=='single_speaker'):
                    wav_temp = wav[smpStart:smpEnd]
                    speech_timestamps1 = get_speech_timestamps(wav_temp, model, threshold=0.9, sampling_rate=SAMPLING_RATE)
                    # print(f'wav_temp={len(wav_temp)} smpStart={smpStart} speech_timestamps1={speech_timestamps1}')
                    
                    for chunks in speech_timestamps1:
                        if chunks['start']>0:
                            chunk_dur = round(chunks['start']/SAMPLING_RATE, 2)
                            with open(re_vad_file, 'a+') as fid:
                                fid.write(f"{smpStart},{smpStart+chunks['start']},others,{chunk_dur}\n")
                            all_chunks += 1
                            
                        chunk_dur = round((chunks['end']-chunks['start'])/SAMPLING_RATE, 2)
                        with open(re_vad_file, 'a+') as fid:
                            fid.write(f"{smpStart+chunks['start']},{smpStart+chunks['end']},single_speaker,{chunk_dur}\n")
                        all_chunks += 1
                else:
                    with open(re_vad_file, 'a+') as fid:
                        fid.write(f"{row['start']},{row['end']},{row['label']},{row['duration']}\n")
                    all_chunks += 1
                    
        # print(f'all_chunks={all_chunks}')
        vad_file = save_dir + '/'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '.csv'
        os.remove(vad_file)
        os.rename(re_vad_file, vad_file)

        vad_textgrid_file = save_dir + '/'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '.textgrid'
        with open(vad_textgrid_file, 'w+') as textgrid_fid:
            textgrid_fid.write('    File type = "ooTextFile"\n')
            textgrid_fid.write('    Object class = "TextGrid"\n')
            textgrid_fid.write('    xmin = 0\n')
            textgrid_fid.write(f'    xmax = {len(wav)/SAMPLING_RATE}\n')
            textgrid_fid.write('    tiers? <exists>\n')
            textgrid_fid.write('    size = 1\n')
            textgrid_fid.write('    item []:\n')
        
        num_lines = 0
        with open(vad_file, 'r') as vad_fid:
            lines = vad_fid.readlines()
            num_lines = len(lines)

        with open(vad_file, 'r') as vad_fid:
            readlines = csv.DictReader(vad_fid)
            with open(vad_textgrid_file, 'a+') as textgrid_fid:
                textgrid_fid.write('       item [1]:\n')
                textgrid_fid.write('          class = "IntervalTier"\n')
                textgrid_fid.write('          name = "speaker_tier"\n')
                textgrid_fid.write("          xmin = 0\n")
                textgrid_fid.write(f"          xmax = {len(wav)/SAMPLING_RATE}\n")
                textgrid_fid.write(f"          intervals: size = {num_lines-1}\n")
            row_count = 1
            for row in readlines:
                with open(vad_textgrid_file, 'a+') as textgrid_fid:
                    textgrid_fid.write(f'          intervals [{row_count}]:\n')
                    textgrid_fid.write(f"                xmin = {float(row['start'])/SAMPLING_RATE}\n")
                    textgrid_fid.write(f"                xmax = {float(row['end'])/SAMPLING_RATE}\n")
                    textgrid_fid.write(f"                text = \"{row['label']}\"\n")
                row_count += 1


re_vad_new()


def remove_new():
    for name in glob.glob(save_dir + '/*.csv' ):
        re_vad_remove_file = save_dir + '/.'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '.csv'
        if os.path.exists(re_vad_remove_file):
            print(f'{name} RE-VAD csv exists')
            continue
        print(re_vad_remove_file)
        
        with open(re_vad_remove_file, 'w+') as fid:
            fid.write('start,end,label,duration\n')

        with open(name, 'r') as fid:
            read_lines = csv.DictReader(fid)
            prev_label = ''
            start = 0
            end = 0
            for row in read_lines:
                rowStart = int(row['start'])
                rowEnd = int(row['end'])
                rowDuration = float(row['duration'])
                rowLabel = row['label']
                if prev_label=='':
                    prev_label = rowLabel
                    end = rowEnd                    
                elif prev_label==rowLabel:
                    end = rowEnd
                elif rowDuration<1:
                    end = rowEnd
                elif not prev_label==rowLabel:
                    if end>start:
                        with open(re_vad_remove_file, 'a+') as fid:
                            fid.write(f"{start},{end},{prev_label},{round((end-start)/SAMPLING_RATE,2)}\n")
                        print(f"{start//SAMPLING_RATE},{end/SAMPLING_RATE},{prev_label},{round((end-start)/SAMPLING_RATE,2)}\n")
                    start = rowStart
                    end = rowEnd
                    prev_label = rowLabel

            if end>start:
                with open(re_vad_remove_file, 'a+') as fid:
                    fid.write(f"{start},{end},{prev_label},{round((end-start)/SAMPLING_RATE,2)}\n")
                # print(f"{start},{end},{prev_label},{round((end-start)/SAMPLING_RATE,2)}\n")
                    

        vad_file = save_dir + '/'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '.csv'
        os.remove(vad_file)
        os.rename(re_vad_remove_file, vad_file)
        
        wav_file = save_dir + '/' + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '.wav'
        wav = read_audio(wav_file, sampling_rate = SAMPLING_RATE)

        vad_textgrid_file = save_dir + '/'  + os.path.splitext(os.path.basename(name))[0].replace(" ", "_") + '.textgrid'
        with open(vad_textgrid_file, 'w+') as textgrid_fid:
            textgrid_fid.write('    File type = "ooTextFile"\n')
            textgrid_fid.write('    Object class = "TextGrid"\n')
            textgrid_fid.write('    xmin = 0\n')
            textgrid_fid.write(f'    xmax = {len(wav)/SAMPLING_RATE}\n')
            textgrid_fid.write('    tiers? <exists>\n')
            textgrid_fid.write('    size = 1\n')
            textgrid_fid.write('    item []:\n')
        
        num_lines = 0
        with open(vad_file, 'r') as vad_fid:
            lines = vad_fid.readlines()
            num_lines = len(lines)

        with open(vad_file, 'r') as vad_fid:
            readlines = csv.DictReader(vad_fid)
            with open(vad_textgrid_file, 'a+') as textgrid_fid:
                textgrid_fid.write('       item [1]:\n')
                textgrid_fid.write('          class = "IntervalTier"\n')
                textgrid_fid.write('          name = "speaker_tier"\n')
                textgrid_fid.write("          xmin = 0\n")
                textgrid_fid.write(f"          xmax = {len(wav)/SAMPLING_RATE}\n")
                textgrid_fid.write(f"          intervals: size = {num_lines-1}\n")
            row_count = 1
            for row in readlines:
                with open(vad_textgrid_file, 'a+') as textgrid_fid:
                    textgrid_fid.write(f'          intervals [{row_count}]:\n')
                    textgrid_fid.write(f"                xmin = {float(row['start'])/SAMPLING_RATE}\n")
                    textgrid_fid.write(f"                xmax = {float(row['end'])/SAMPLING_RATE}\n")
                    textgrid_fid.write(f"                text = \"{row['label']}\"\n")
                row_count += 1


# remove_new()
        
