from PIL import Image, ImageChops
import numpy as np
import os

from tqdm import tqdm

path_Market = "/datasets_nas/ange8547/Market-1501-v15.09.15/"
path_DNM = "/datasets_nas/ange8547/Market-DNet/"


"""
img1 = Image.open(os.path.join(path_Market, "bounding_box_test/0001_c1s1_001051_03.jpg")).resize((128, 256), Image.BILINEAR)
img2 = Image.open(os.path.join(path_DNM, "bounding_box_test/0001_c1s1_001051_03.jpg"))

diff = ImageChops.difference(img1, img2)
#diff_arr = np.array(diff)
#max_scale = 255 / np.max(diff_arr)
#diff = Image.fromarray((diff_arr * 255 / np.max(diff_arr)).astype(np.uint8))

diff.save(f"DNM_diff/test.jpg")
exit()"""

subsets = ["bounding_box_train", "bounding_box_test", "query"]

image_size = (0,0)
image_size_set = False

summaries = []


for set in tqdm(subsets):

    min_diff_pixel_count = 999999999999999999
    min_diff_path = ""

    max_diff_pixel_count = 0
    max_diff_path = ""

    total_different_pixels = 0
    pixel_differences = []

    total_different_images = 0

    total_images = 0

    pixel_difference_averages = []
    nz_pixel_difference_averages = []

    for filename in tqdm(os.listdir(os.path.join(path_DNM, set))):

        total_images += 1

        img1 = Image.open(os.path.join(path_Market, set, filename)).resize((128, 256), Image.BILINEAR)
        img2 = Image.open(os.path.join(path_DNM, set, filename))

        diff = ImageChops.difference(img1, img2)

        diff_arr = np.array(diff)

        if not image_size_set:
            image_size = (diff_arr.shape[0], diff_arr.shape[1])
            image_size_set = True

        pixel_difference = np.count_nonzero(diff_arr)

        total_different_pixels += pixel_difference

        if pixel_difference < min_diff_pixel_count:
            min_diff_pixel_count = pixel_difference
            min_diff_path = os.path.join(set, filename)

        if pixel_difference > max_diff_pixel_count:
            max_diff_pixel_count = pixel_difference
            max_diff_path = os.path.join(set, filename)

        if pixel_difference > 0:
            total_different_images += 1
            pixel_differences.append(pixel_difference)


        # Calculate the average value of the difference image per channel
        avg_diff_per_channel = np.mean(diff_arr, axis=(0, 1))

        # Calculate the average value of the non-zero entries in the difference image per channel
        non_zero_diff_arr = diff_arr[diff_arr != 0]
        avg_non_zero_diff_per_channel = np.mean(non_zero_diff_arr, axis=0)

        pixel_difference_averages.append(avg_diff_per_channel)
        nz_pixel_difference_averages.append(avg_non_zero_diff_per_channel)
    
    # Convert the list of arrays into a single NumPy array
    pixel_difference_averages_arr = np.array(pixel_difference_averages)

    # Calculate the average over the list
    average_pixel_difference = np.mean(pixel_difference_averages_arr, axis=0)

    # Convert the list of arrays into a single NumPy array
    nz_pixel_difference_averages_arr = np.array(nz_pixel_difference_averages)

    # Calculate the average over the list
    nz_average_pixel_difference = np.mean(nz_pixel_difference_averages_arr, axis=0)

    # Convert the list of arrays into a single NumPy array
    pixel_differences_arr = np.array(pixel_differences)

    # Calculate the average over the list
    avg_num_differing_pixels = np.mean(pixel_differences_arr, axis=0)

    summary  = f"For the subset {set}, {total_different_images}/{total_images} or {round(total_different_images/total_images*100, 2)}% of images were different.\n"
    summary += f"On average, {round(avg_num_differing_pixels, 2)} or {round(avg_num_differing_pixels/(image_size[0]*image_size[1])*100/3, 2)}% pixels of a given image were different.\n"
    summary += f"On average, they differed by {round(nz_average_pixel_difference, 2)} and the total average difference per pixel of an image was {average_pixel_difference}.\n"
    summary += f"The image with the most difference ({max_diff_pixel_count} different pixels) was {max_diff_path} and the one with the least difference ({min_diff_pixel_count} different pixels) was {min_diff_path}. \n"
    
    if min_diff_path != "":
        img1 = Image.open(os.path.join(path_Market, min_diff_path)).resize((128, 256), Image.BILINEAR)
        img2 = Image.open(os.path.join(path_DNM, min_diff_path))

        diff = ImageChops.difference(img1, img2)
        diff_arr = np.array(diff)
        min_scale = 255 / np.max(diff_arr)
        diff = Image.fromarray((diff_arr * min_scale).astype(np.uint8))

        diff.save(f"{set}_min.jpg")

    if max_diff_path != "":
        img1 = Image.open(os.path.join(path_Market, max_diff_path)).resize((128, 256), Image.BILINEAR)
        img2 = Image.open(os.path.join(path_DNM, max_diff_path))

        diff = ImageChops.difference(img1, img2)
        diff_arr = np.array(diff)
        max_scale = 255 / np.max(diff_arr)
        diff = Image.fromarray((diff_arr * max_scale).astype(np.uint8))

        diff.save(f"{set}_max.jpg")
    summary += f"The max diff image was scaled by {max_scale} and the min diff image was scaled by {min_scale}.\n"

    summaries.append(summary)

        
for summary in summaries:
    print(summary)


