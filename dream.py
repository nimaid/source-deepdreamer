from __future__ import print_function

import numpy as np
import PIL.Image

import tensorflow as tf

import random, os

model_fn = 'tensorflow_inception_graph.pb'
# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

layer_names = ['mixed5b', 'mixed5a', 'mixed4e', 'mixed4d', 'mixed4c', 'mixed4b', 'mixed4a', 'mixed3b', 'mixed3a']

img_noise = np.random.uniform(size=(224,224,3)) + 100.0

def array_to_img(a):
    a = np.uint8(np.clip(a, 0, 1)*255)
    return PIL.Image.fromarray(a)

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def open_pic(pic_name):
    pic = PIL.Image.open(pic_name)
    return pic

def pic_to_array(pic):
    return np.float32(pic)

def open_pic_as_array(pic_name):
    pic = open_pic(pic_name)
    return pic_to_array(pic)

def render_deepdream(t_obj, img0=img_noise,
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('.',end = '')
        print('Octave {}/{} done.'.format(octave + 1, octave_n))
    
    final_pic = array_to_img(img/255.0)
    return final_pic

def random_deepdream(img=img_noise,
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    layer_name = random.choice(layer_names)
    t_obj = tf.square(T(layer_name))
    return render_deepdream(t_obj, img, iter_n, step, octave_n, octave_scale)
    
def random_deepdream_folder(in_folder, out_folder=None, iter_n=10):
    if out_folder == None:
        out_folder = in_folder + '_dreamed'

    in_folder = os.path.normpath(in_folder)
    out_folder = os.path.normpath(out_folder)
    
    for root, dirs, files in os.walk(in_folder):
        for name in files:
            open_path = os.path.join(root, name)
            
            save_root = root.replace(in_folder, out_folder)
                
            save_path = os.path.join(save_root, name)

            save_dir = os.path.dirname(save_path)
                
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            special = None
            for special_type in ['normal', 'exponent', 'occlusion', 'mask', 'phong', 'dudv']:
                if special_type + os.path.splitext(name)[1] in name:
                    special = special_type
                    break

            print('Processing image "{}"'.format(open_path))
            
            if special != None:
                open_pic(open_path).save(save_path)
                print('Coppied unmodified {} image to "{}"'.format(special, open_path))
            elif os.path.exists(save_path):
                print('Skipped already dreamed image "{}"'.format(save_path))
            else:
                print('Dreaming about the image...')
                try:
                    temp_img = open_pic_as_array(open_path)

                    temp_dream = random_deepdream(temp_img, iter_n=iter_n)

                    temp_dream.save(save_path)
                    del temp_img
                    del temp_dream
                    print('Dreamed to "{}"'.format(save_path))
                except KeyboardInterrupt:
                    print("Stopping conversion...")
                    return
                except:
                    open_pic(open_path).save(save_path)
                    print('ERROR: Image not modified. Coppied unmodified image to "{}"'.format(open_path))
            print("")

random_deepdream_folder("C:\\Users\\ellag\\Documents\\HL2 Modding\\Deep Dream\\hl2_textures_dir_raw_bmp")
