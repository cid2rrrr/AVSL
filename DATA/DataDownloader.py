import csv
import pandas as pd
import os
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
from moviepy.video.io.VideoFileClip import VideoFileClip


CSV_PATH = './unbalanced_train_segments.csv'
OUTPUT_PATH = './videos' # need to change to server dir to save the clips


def download_and_trim_video(cnt, youtube_id, output_folder, start_time_seconds, end_time_seconds):
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
        clip = VideoFileClip(video_path).subclip(float(start_time_seconds), float(end_time_seconds))

        # Generate a unique name for the trimmed clip based on the video title
        # output_clip_name = f"{video_title.replace(' ', '_')}_trimmed.mp4"
        # output_clip_name = str(youtube_id)+"_trimmed.mp4"
        output_clip_name = str(cnt)+".mp4"
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



def main():
    csv_list=[]

    f=open(CSV_PATH, encoding="utf8") # need to change to server csv file path
    reader = csv.reader(f)

    for l in reader:
        csv_list.append(l)
    f.close()
    del csv_list[0]
    del csv_list[0]
    del csv_list[0]

    for cnt, entry in enumerate(csv_list):
        files = os.listdir(OUTPUT_PATH)
        if str(cnt)+'.mp4' in files:
            print(str(cnt)+".mp4 Already Exists")
        else:
            youtube_id, start_time, end_time, *_ = entry
            download_and_trim_video(cnt, youtube_id, OUTPUT_PATH, start_time, end_time)


if __name__ == "__main__":
    main()
