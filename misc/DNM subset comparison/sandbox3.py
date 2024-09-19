from PIL import Image, ImageChops
import numpy as np
import os

from tqdm import tqdm

path01 = "/datasets_nas/ange8547/Market-DNet/bounding_box_train_0.1_0/"
path = "/datasets_nas/ange8547/Market-DNet/bounding_box_train/"

print(f"Question: Are there any different files between the two folders '{path}' and '{path01}'?")

ctr = 0
nzp = 0

for filename in tqdm(os.listdir(path01)): # they have the same names in both paths
    # Read in the PNG file
    img1 = Image.open(os.path.join(path01, filename))
    img2 = Image.open(os.path.join(path, filename))

    diff = ImageChops.difference(img1, img2)

    diff_arr = np.array(diff)

    nzp += np.count_nonzero(diff_arr)

    ctr += 1

print(f"Checked {ctr}/12936 images and found {nzp} differing pixels.")
