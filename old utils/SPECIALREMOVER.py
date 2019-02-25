# I only used this to purge unchanged assets when I just copied them like a dumb
# You won't need this.

import os

def remove_specials(in_folder):
    in_folder = os.path.normpath(in_folder)
    
    for root, dirs, files in os.walk(in_folder):
        for name in files:
            open_path = os.path.join(root, name)
            
            special = None
            for special_type in ['normal', 'exponent', 'occlusion', 'mask', 'phong', 'dudv', 'spec']:
                if special_type + os.path.splitext(name)[1] in name:
                    special = special_type
                    break
            
            if special != None:
                os.remove(open_path)
                print('Removed {} image "{}"'.format(special, open_path))

remove_specials("C:\\Users\\ellag\\Documents\\HL2 Modding\\Deep Dream\\deepdream")
