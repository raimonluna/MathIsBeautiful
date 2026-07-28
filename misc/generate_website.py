import os
import re
import numpy as np
from glob import glob

######################### RETRIEVE INFORMATION #########################

image_path    = "../website/images/"
movie_path    = "../website/movies/"
image_names   = ['images/' + name.split("/")[-1] for name in sorted(glob(f'{image_path}/MIB*.jpg'))]
movie_names   = [name.split("/")[-1] for name in sorted(glob(f'{movie_path}/MIB*.mp4'))]
captions_dirs = sorted(glob('../MIB*/**/*.txt', recursive = True))

captions = []
for caption_file in captions_dirs:
    with open(caption_file) as f:
        captions += f.read().split(65 * '=')
captions = [c.strip('\n') for c in captions]
captions = [c for c in captions if len(c) > 0]

############################ WRITE THE FILE ############################

def make_button(index):

    image_name = media_name = image_names[index]
    caption = captions[index].replace('"', '&quot;').replace("'", '&apos;').split('\n\n')
    
    if image_name.split('/')[-1][:-4] + ".mp4" in movie_names:
        media_name = 'movies/' + image_name.split('/')[-1][:-4] + ".mp4"
    title = re.sub(r"(\w)([A-Z])", r"\1 \2", image_name.split('/')[-1][:-4].split('/')[-1][8:])
    text  = '\n' + '\n'.join([f'          data-p{i + 1}="' + caption[i] + '"' for i in range(len(caption) - 1)])
    url   = caption[-1][29:]

    button = f"""
        <button class="mb-tile"
          data-thumb="images/math-beautiful/{image_name}"
          data-full="images/math-beautiful/{media_name}"
          data-date="Math is Beautiful"
          data-title="{title}"{text}
          data-repo="{url}">
          <img src="images/math-beautiful/{image_name}" alt="Template animation poster frame">
          <span class="mb-cap">{title}</span>
        </button>
    """
    return button

os.remove("mb-data.html")
with open("mb-data.html", "a") as f:
    for index in range(len(image_names)):
        f.write(make_button(index))

