import os, shutil

def folderthing(in_folder):
    in_folder = os.path.normpath(in_folder)
    for root, dirs, files in os.walk(in_folder):
        for name in files:
            open_path = os.path.join(root, name)

            if os.path.splitext(open_path)[1] != '.vtf':
                os.remove(open_path)
            
