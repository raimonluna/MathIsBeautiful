#!/bin/bash

in_path="/home/raimon/dev/MathIsBeautiful/Output"
out_img="/home/raimon/dev/MathIsBeautiful/website/images"
out_mov="/home/raimon/dev/MathIsBeautiful/website/movies"

# Ensure output directories exist
mkdir -p "$out_img"
mkdir -p "$out_mov"

# Save images in jpg format
for f in "$in_path"/Images/*.png; do
    [ -e "$f" ] || continue
    filename=$(basename "$f")
    convert "$f" -resize 1024x "$out_img/${filename%.png}.jpg"
done

# Compress the movies (with size check fallback)
for f in "$in_path"/Movies/*.mp4; do
    [ -e "$f" ] || continue
    filename=$(basename "$f")
    
    echo "Processing video: $filename..."
    
    # Create a temporary file for the compression attempt
    temp_output="$out_mov/${filename%.mp4}.tmp.mp4"
    
    ffmpeg -nostdin -i "$f" -vcodec libx264 -crf 32 -preset faster -tune animation -vf "scale='min(1280,iw)':-2" -acodec aac -b:a 64k "$temp_output" -y
    
    # Get file sizes in bytes
    original_size=$(stat -c%s "$f")
    compressed_size=$(stat -c%s "$temp_output")
    
    # Check if the compressed version is actually smaller
    if [ "$compressed_size" -lt "$original_size" ]; then
        echo "Success: Compressed video is smaller ($compressed_size vs $original_size bytes). Keeping compressed version."
        mv "$temp_output" "$out_mov/$filename"
    else
        echo "Warning: Compressed video is larger or equal ($compressed_size vs $original_size bytes). Copying original instead."
        rm "$temp_output"
        cp "$f" "$out_mov/$filename"
    fi
done

# Make video thumbnails (overwrite existing images)
while read -r timestamp video; do
    # Strip any hidden Windows carriage returns (\r) and skip empty lines
    timestamp=$(echo "$timestamp" | tr -d '\r')
    video=$(echo "$video" | tr -d '\r')
    
    if [ -z "$timestamp" ] || [ -z "$video" ]; then
        continue
    fi
    
    framename="${video%.mp4}.jpg"
    echo "Extracting frame from $video at $timestamp..."
    
    # FIXED: -i comes BEFORE -ss to prevent the ffmpeg freezing error
    ffmpeg -nostdin -i "$in_path"/Movies/"$video" -ss "$timestamp" -frames:v 1 "$out_img"/"$framename" -y
done < "thumbnail_list.txt"

echo "Process completed successfully!"

