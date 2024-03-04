import os

def get_file_paths(song_names, base_folder):
    video_paths = []
    mixed_audio_paths = []
    separated_audio_paths = []

    for root, dirs, files in os.walk(base_folder):
        for dir_name in dirs:
            for song_name in song_names:
                if song_name in dir_name:
                    song_folder = os.path.join(root, dir_name)
                    for root, dirs, files in os.walk(song_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            base_file_name, file_extension = os.path.splitext(file)
                            if file_extension.lower() == ".mp4":
                                video_paths.append(file_path)
                            elif "AuMix" in base_file_name:
                                mixed_audio_paths.append(file_path)
                            elif "AuSep" in base_file_name:
                                separated_audio_paths.append(file_path)
    
    return video_paths, mixed_audio_paths, separated_audio_paths

# # Example usage:
# song_names = ["{:02d}".format(i) for i in range(1, 45)]
# base_folder = "/workspace/GitHub/AVSL/DATA/Dataset"

# video_paths = []
# mixed_audio_paths = []
# separated_audio_paths = []
# for song_name in song_names:
#     video_paths_temp, mixed_audio_paths_temp, separated_audio_paths_temp = get_file_paths([song_name], base_folder)
#     video_paths.extend(video_paths_temp)
#     mixed_audio_paths.extend(mixed_audio_paths_temp)
#     separated_audio_paths.extend([separated_audio_paths_temp])

# print("Video Paths:")
# print(video_paths)
# print("\nMixed Audio Paths:")
# print(mixed_audio_paths)
# print("\nSeparated Audio Paths:")
# print(separated_audio_paths)