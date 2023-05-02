1) create a conda environment (if don't have conda install https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)

example: conda create -n mlsv_youtube python=3.8


2) Go to the downloaded folder through terminal and run: pip install -r requirements.txt
  
4) conda install -c conda-forge ffmpeg

5) Once installed open the 'run_script.sh' file, present in 'Crawl_youtube' and edit the path and threshold value, after carefullyreading the instruction and run the same script:
bash run_script.sh


******************************
Important information:
******************************

Youtube now requires logging into gmail account to download videos. When you first run the script, you wil see the following lines in the terminal:

		$ Please open https://www.google.com/device and input code <CODE SHOWN IN TERMINAL>
		$ Press enter when you have completed this step.

When you see this, go to https://www.google.com/device and provide the code shown in the terminal at the appropriate place in the webpage when prompted. You may need to sign-in to your gmail account. Once this step is done, go to the terminal and press Enter. Your videos should start downloading. 




