# I used this for when I accidentally used BMP
# You won't need this

import os, PIL.Image

def alpha_transfer_folders(in_folder, out_folder):
    in_folder = os.path.normpath(in_folder)
    out_folder = os.path.normpath(out_folder)
    
    for root, dirs, files in os.walk(in_folder):
        for name in files:
            open_path = os.path.join(root, name)
            
            save_root = root.replace(in_folder, out_folder)
                
            save_path = os.path.join(save_root, os.path.splitext(name)[0] + '.bmp')

            save_dir = os.path.dirname(save_path)
                
            
            print('Processing image "{}"'.format(open_path))
            print(save_path)
            if os.path.exists(save_path):
                print('Destination exists, copying alpha...')
                img = PIL.Image.open(open_path)
                img.load()
                if len(img.getbands()) >= 4:
                    #get alpha of source
                    img_split = img.split()
                    img_a = img_split[3]

                    #get rgb of destination
                    img2 = PIL.Image.open(save_path)
                    img2_rgb = list(img2.split())[:3]

                    img_new = PIL.Image.merge('RGBA', img2_rgb + [img_a])

                    img_new_ext = os.path.splitext(open_path)[1]
                    img_new_name = os.path.splitext(save_path)[0] + img_new_ext

                    img_new.save(img_new_name)
                    print('Saved to "{}"!'.format(img_new_name))
            print('')

alpha_transfer_folders("C:\\Users\\ellag\\Documents\\HL2 Modding\\Deep Dream\\hl2_textures_dir_png", "C:\\Users\\ellag\\Documents\\HL2 Modding\\Deep Dream\\hl2_textures_dir_bmp_dreamed")

