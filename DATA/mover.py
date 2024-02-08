import os, shutil, re

pattern = re.compile(r'\D')

for filename in os.listdir('./videos/'):
    if filename.endswith('.mp4'):
        if re.search(r'[^[0-9]', filename.split('.')[0]):
            shutil.move(os.path.join('./videos/', filename), os.path.join('./err_videos/', filename))
