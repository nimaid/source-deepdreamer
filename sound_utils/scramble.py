import random, os, shutil

def files_in_subdirs(in_folder):
    return sum([len(files) for r, d, files in os.walk(in_folder)])

def scramble_folder(in_folder, out_folder=None):
    if out_folder == None:
        out_folder = in_folder + '_scrambled'

    in_folder = os.path.normpath(in_folder)
    out_folder = os.path.normpath(out_folder)

    subdirs = []

    for root, dirs, files in os.walk(in_folder):
        for name in files:
            path = os.path.join(root, name)
            path = path.replace(in_folder + os.path.sep, '')
            subdirs.append(path)

    dest_subdirs = subdirs.copy()
    random.shuffle(dest_subdirs)

    total_files = files_in_subdirs(in_folder)
    num_files = 0
    for idx, source_subpath in enumerate(subdirs):
        source = os.path.join(in_folder, source_subpath)
        dest = os.path.join(out_folder, dest_subdirs[idx])
        
        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        shutil.copyfile(source, dest)

        num_files += 1
        if num_files % 10 == 0:
            print('{}/{}'.format(num_files, total_files))
