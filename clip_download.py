import csv
import pandas as pd
import cv2
import torch
import os
import librosa
import numpy as np
import subprocess
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
from moviepy.video.io.VideoFileClip import VideoFileClip
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

f=open('unbalanced_train_segments.csv', encoding="utf8") # need to change to server csv file path
reader = csv.reader(f)
csv_list=[]
for l in reader:
  csv_list.append(l)
f.close()
#log_df = pd.DataFrame(csv_list)
#log_df
del csv_list[0]
del csv_list[0]
del csv_list[0]
# url = 'https://youtu.be/'+csv_list[i][0]
# start_sec = int(csv_list[i][1].split()[0])
# end_sec = int(csv_list[i][2].split()[0])

def download_and_trim_video(youtube_id, output_folder, start_time_seconds, end_time_seconds):
    # YouTube video URL with the specified ID
    video_url = f"https://www.youtube.com/watch?v={youtube_id}"

    try:
        # Create a YouTube object
        youtube = YouTube(video_url)

        # Get the highest resolution stream (you can choose a different stream if needed)
        video_stream = youtube.streams.get_highest_resolution()

        # Specify the output path for downloading the video
        video_path = video_stream.download(output_folder)

        # Get the video title to use as a base for the output file name
        video_title = youtube.title

        # Trim the video clip
        clip = VideoFileClip(video_path).subclip(start_time_seconds, end_time_seconds)

        # Generate a unique name for the trimmed clip based on the video title
        output_clip_name = f"{video_title.replace(' ', '_')}_trimmed.mp4"
        output_clip_path = os.path.join(output_folder, output_clip_name)

        # Save the trimmed clip
        clip.write_videofile(output_clip_path, codec="libx264", audio_codec="aac")
        clip.close()

        # Delete the full-length video
        os.remove(video_path)
    
    # Deleted video exception code
    except VideoUnavailable:
        print(f"Video with YouTube ID {youtube_id} is unavailable or has been deleted.")
    
    except Exception as e :
        print(f"Error processing video with YouTube ID {youtube_id}: {e}")

output_path = 'output/path' # need to change to server dir to save the clips
for entry in csv_list:
    youtube_id, start_time, end_time, *_ = entry
    download_and_trim_video(youtube_id, output_path, start_time, end_time)