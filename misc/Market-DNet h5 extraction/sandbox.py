import h5py
import numpy as np
from PIL import Image
import os


in_path = '/datasets_nas/ange8547/Market/'
out_path = '/datasets_nas/ange8547/Market-DNet/'
#sets = ['bounding_box_train', 'bounding_box_test', 'query']
sets = ['bounding_box_train_0.1_0']

for set in sets:
    with h5py.File(os.path.join(in_path, set+".h5"), 'r') as f:
        for img_array, name in zip(f["image_data"], f["image_name"]):
            
            filename = str(name).split("/")[-1][:-2]
            assert filename[-4:] == ".jpg"

            image = Image.fromarray(np.uint8(img_array))

            image.save(os.path.join(out_path, set, filename))


exit()
# Open the h5 file
with h5py.File('/datasets_nas/ange8547/Market/bounding_box_train.h5', 'r') as f:
    # Get the dataset containing the images
    """ testing
    print(f.keys())
    for k in f.keys():
        print(k + ":", f[k][0])
    print(str(f["image_name"][0]).split("/")[-1][:-2])# get image name

    print(f["image_data"].__class__.__name__)
    print(f["image_data"][0].__class__.__name__) # ndarray
    print(f["image_data"][0].shape) # (256, 128, 3)

    #image = Image.fromarray(np.uint8(f["image_data"][0]))
    #image.save("z_test.jpg") # image looks fine
    """
    


    #for img_array, name in zip(f["image_data"], f["image_name"]):


    exit()
    
# Save each image as a file
"""for i in range(images.shape[0]):
    image_data = images[i].transpose(1, 2, 0)
    image = Image.fromarray(np.uint8(image_data))
    image.save('image_{}.jpg'.format(i))"""