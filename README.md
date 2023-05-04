1) create a conda environment (if don't have conda install https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)

example: conda create -n mlsv_youtube python=3.8


2) Go to the downloaded folder through terminal and run: pip install -r requirements_pip.txt
  
4) conda install -c conda-forge ffmpeg

5) Once installed open the 'run_script.sh' file, present in the current directory and edit the path and threshold value, after carefully reading the instruction and run the same script:
bash run_script.sh


******************************
Important information:
******************************

Youtube now requires logging into gmail account to download videos. When you first run the script, you wil see the following lines in the terminal:

		$ Please open https://www.google.com/device and input code <CODE SHOWN IN TERMINAL>
		$ Press enter when you have completed this step.

When you see this, go to https://www.google.com/device and provide the code shown in the terminal at the appropriate place in the webpage when prompted. You may need to sign-in to your gmail account. Allow "Youtube on TV" access to your gmail account. Once this step is done, go to the terminal and press Enter. Your videos should start downloading. 


6) After running the script, two directories will be created one level above the current working directory. 
	1-> Videos  : Contains .mp4, and .mp3 files of the downloaded videos
	2-> Textgrid: Contains .wav, .csv and .textgrid files of the downloaded videos.
	
	Please note that the Textgrid directory will contain two .textgrid files for each video. 
	
	One will be named as <video_id>.textgrid, while the other will be named <video_id>_thresholded.textgrid.
		<video_id>.textgrid contains the annotations obtained just after performing VAD
		<video_id>_thresholded.textgrid contains the annotations obtained after performing cosine similarity based filtering single speaker speech. You are supposed to use this textgrid file for updating the annotations using Praat. 
	
	

