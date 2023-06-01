import os
import glob

def count_lines_in_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return len(lines)

def count_lines_in_py_files(directory):
    total_lines = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            if file.endswith(".py"):
                total_lines += count_lines_in_file(os.path.join(dirpath, file))
    return total_lines

# Change the directory path as needed. '.' refers to the current directory.
print(count_lines_in_py_files('.'))