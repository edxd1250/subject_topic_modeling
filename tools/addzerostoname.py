import os

# Specify the directory where your files are located
directory = '/path/to/your/directory'

# Get the list of files in the directory
files = os.listdir(directory)

# Sort the files based opan their numerical order
files.sort(key=lambda x: int(x.split('_')[0]))

# Rename the files with leading zeros
for i, file_name in enumerate(files[:99], start=1):
    new_name = f"{i:03d}_ordered_embeddings_scibert.pickle"
    old_path = os.path.join(directory, file_name)
    new_path = os.path.join(directory, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed: {file_name} to {new_name}")
