# Steep Source games in LSD!

This is a little script that uses TensorFlow (not Caffe!) to deep dream all the textures in a folder, while skipping functional textures like normal, exponent, and occlusion maps.

* You must use a VPK extractor (like the Python `vpk` module) to get the folder of VTF files.
* Next, use VTFEdit to batch convert the VTFs into BMPs.
* Run `dream.py` in an interactive python console (`python -i dream.py`)
* Run the function `random_deepdream_folder("folder/path/here")`, or `random_deepdream_folder("folder\\path\\here")` for Windows.
* This will make a new folder and walk through all textures, then putting them in the correct folder structure after being processed.
* Finally, use VTFEdit to batch convert the new BMPs into VTF files again.

Yay! Now just copy the 'materials' folder to the base folder for your game (such as the `hl2` folder where `media`, `models`, etc. live).
Pray and start the game.

You can also pass `out_folder` and `iter_n` arguments to `random_deepdream_folder`
* `out_folder` lets you manually choose a folder path to save to.
* `iter_n` changes how many iterations of dreaming it does. Default is 10, more means trippier, but takes more time to process.