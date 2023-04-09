#!/bin/bash

# Path to folder containing input text files
input_folder="./format_folder"

# Path to output folder
output_folder="./format_folder_out"

# Path to Python script
script_path="./preprocess.py"

# Loop through all files in the input folder
for input_file in $input_folder/*.txt; do

    # Get the filename without extension
    filename=$(basename -- "$input_file")
    filename="${filename%.*}"

    # Run the Python script on the input file and save output to output folder
    python $script_path $input_file $output_folder/$filename"_out.txt"

done
