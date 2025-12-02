# take an argument: name of the file in the Images folder
INPUT_FILE=$1

# set output file name
OUTPUT_FILE="$1-optimized.jpg"

# use imagemagick to resize the image's largest dimension to 1500px if bigger, and optimize quality to 85%
convert "Images/$INPUT_FILE" -resize 1500x1500\> -quality 85 "Images/$OUTPUT_FILE"