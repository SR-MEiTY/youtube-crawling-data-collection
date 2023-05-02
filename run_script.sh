#!/bin/bash
## This propgrame is written to split the youtube downloaded utterances to small chunks of monospeaker utterances of 3-10 sec duration
## Written by Jagabandhu Mishra, 27/07/2022

# Modified by Mrinmoy Bhattacharjee, March 16, 2023

#url_playlist='https://youtube.com/playlist?list=PLAoPcoSH8Y5xVHRBpc6DJgPYYSNnVxRMU' # you can only add the url of the play list here
url_playlist='https://www.youtube.com/watch?v=1Ov4nczAPKo&list=PLw0OS4SJWbYBjpgcIQvCWyaffXd6w_DG4'
#url_playlist='https://www.youtube.com/playlist?list=PLrmkvdIAaxvu5PwR-hfTuUeKBU6LPuX1W'
full_audio_save='../Videos' # provide the path where you want to save
VAD_split_dir='../Textgrids' # Provide the path where want to store the vad split segments
threshold=-1.0 # specify the threshold here, irrelevant / noisy audio files have to be removed. For each language, we have to listen to some audio
#files to define a threshold.All audio files having the threshold value below the pre-defined threshold will be removed
stage=0 # change the stage as per your choice


#if [ $stage -le 0 ]; then
#
#	if [[ ! -d $VAD_split_dir ]]
#	then
#	    echo "$DIRECTORY does not exists: creating."
#	    mkdir $VAD_split_dir
#	fi
#
#
#	if [[ ! -d $full_audio_save ]]
#	then
#	    echo "$DIRECTORY does not exists: creating."
#	    mkdir $full_audio_save
#	fi
#
#fi



#1. Crawl MP4 video from Youtube and convert to WAV:
if [ $stage -le 1 ]; then
python crawl.py --url_playlist=$url_playlist --save_dir=$full_audio_save --vad_dir=$VAD_split_dir
fi


#2. Data Pre-processing (split the audio files into smaller files using Silero Voice Activity Detection (VAD))
if [ $stage -le 2 ]; then
python silero-VAD.py --url_playlist=$url_playlist --video_folder=$full_audio_save --save_dir=$VAD_split_dir
fi



#3.After performing VAD, compute the cosine similarity of audio pairs:
if [ $stage -le 3 ]; then
python cosine_pair.py --url_playlist=$url_playlist --video_dir=$full_audio_save --vad_dir=$VAD_split_dir --threshold=$threshold
fi

