import os

# define the maximum number of data points per file
MAX_DATA_PER_FILE = 100

# define the path to the format_data file and the format_folder directory
DATA_FILE_PATH = 'format_data.txt'
OUTPUT_DIR_PATH = 'format_folder'

# create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR_PATH):
    os.makedirs(OUTPUT_DIR_PATH)

# read the format_data file
with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
    data = f.read()

# split the data into individual data points
data_points = data.split('<s>')[1:]

# create a list of batches, each with up to MAX_DATA_PER_FILE data points
batches = [data_points[i:i + MAX_DATA_PER_FILE] for i in range(0, len(data_points), MAX_DATA_PER_FILE)]

# write each batch to a separate file in the output directory
for i, batch in enumerate(batches):
    # create the file name for this batch
    file_name = os.path.join(OUTPUT_DIR_PATH, f'format_data_{i + 1}.txt')

    # write the data points to the file
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(''.join(['<s>' + dp for dp in batch]))

    print(f'Saved batch {i + 1} to {file_name}')
