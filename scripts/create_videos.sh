#!/bin/bash
CURRENT_DIR=$(pwd)
cd "$1"
for f in *; do
    echo $f 
    ffmpeg -framerate 10 -pattern_type glob -i "${f}/*.png" -c:v libx264 -r 30 -pix_fmt yuv420p "${f}/${f}_out.mp4"
    echo $f
    echo "Done"
done

cd $CURRENT_DIR
