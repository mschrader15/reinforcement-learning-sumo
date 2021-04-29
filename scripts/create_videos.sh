#!/bin/bash
COUNTER=0
CURRENT_DIR=$(pwd)
cd "$1"
for f in *; do
    echo $f
    # if [ -d "${f}" ]; then
        # checkpoint=$(dirname -- "$f/nano.txt")
        # echo $checkpoint
        # echo $f
    let COUNTER=COUNTER+1 
    ffmpeg -framerate 5 -pattern_type glob -i "${f}/*.png" -c:v libx264 -r 30 -pix_fmt yuv420p "${f}/${f}_out.mp4"
    echo $f
    echo "Done"
    # fi
done

cd $CURRENT_DIR
