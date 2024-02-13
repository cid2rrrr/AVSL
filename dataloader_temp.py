import csv
import pandas as pd
import os
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
from moviepy.video.io.VideoFileClip import VideoFileClip

# parallel  
from subprocess import Popen, PIPE
from multiprocessing import Pool


CSV_PATH = './unbalanced_train_segments.csv'
OUTPUT_PATH = './videos' # need to change to server dir to save the clips
NUM_CONTAINERS = 10 # number of containers

def download_and_trim_video(cnt, youtube_id, start_time_seconds, end_time_seconds):
    video_url = f"https://www.youtube.com/watch?v={youtube_id}"
    output_clip_name = f"{cnt}.mp4"
    output_clip_path = os.path.join(OUTPUT_PATH, output_clip_name)

    # The docker run command is used to execute the download_and_trim.py script
    # in a Docker container, passing the necessary arguments
    command = f'docker run --rm -v "$(pwd)/videos:/videos" our_download_container python download_and_trim.py {youtube_id} {start_time_seconds} {end_time_seconds} {output_clip_path}'

    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    if stderr:
        print(f"Error downloading and trimming video {youtube_id}: {stderr.decode('utf-8')}")
    else:
        print(f"Video {youtube_id} downloaded and trimmed successfully.")


def main():
    csv_list = []

    with open(CSV_PATH, encoding="utf8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        next(reader)  # Skip another line
        next(reader)  # Skip another line

        for row in reader:
            csv_list.append(row)

    with Pool(processes=NUM_CONTAINERS) as pool:
        for cnt, entry in enumerate(csv_list):
            youtube_id, start_time, end_time, *_ = entry
            pool.apply_async(download_and_trim_video, (youtube_id, start_time, end_time, cnt))

        pool.close()
        pool.join()


if __name__ == "__main__":
    main()

# download_and_trim.py
"""
import sys
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip

def download_and_trim(cnt, youtube_id, output_folder, start_time_seconds, end_time_seconds):
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


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python download_and_trim.py youtube_id start_time_seconds end_time_seconds output_path")
        sys.exit(1)

    youtube_id = sys.argv[1]
    start_time_seconds = sys.argv[2]
    end_time_seconds = sys.argv[3]
    output_path = sys.argv[4]

    download_and_trim(youtube_id, start_time_seconds, end_time_seconds, output_path)
"""