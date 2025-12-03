# take an argument: name of the file in the Images folder
Set-Variable -Name INPUT_FILE -Value $args[0]

# set output file name (cut the input file extension and add -optimized.jpg)
$OUTPUT_FILE = "$([System.IO.Path]::GetFileNameWithoutExtension($args[0]))-optimized-hard.jpg"

# # use imagemagick to resize the image's largest dimension to 1500px if bigger, and optimize quality to 85%
magick ".\Images\$INPUT_FILE" -resize 1200x1200\> -quality 85 "Images/$OUTPUT_FILE"