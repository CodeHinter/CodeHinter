import os

# Set the directory containing the text files
dir_path = './format_folder_out'

# Open a new file for writing
with open('output.txt', 'w') as outfile:
    # Iterate over all the text files in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            # Open each file and read its contents
            with open(os.path.join(dir_path, filename), 'r') as infile:
                contents = infile.read()

                # Write the contents to the output file
                outfile.write(contents)