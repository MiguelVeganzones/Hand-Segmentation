from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random


from read_images import img_size, INPUT_IMG_DIR, TARGET_IMG_DIR, WEIGHT_MAPS_DIR, get_dataset_paths
from train import SEED1, N, train_samples

"""
Random cropping
Small rotations
Horizontal flipping
Saturation and color changes
"""

storepath = "../../../annotations/egohands/egohands_data/_AUGMENTED_SAMPLES/"
x_store_path = "_LABELLED_SAMPLES/"
y_store_path = "_GROUND_TRUTH_DISJUNCT_HANDS_2/"
wm_store_path = "_WEIGHT_MAPS_08_03_22/"

def horizontal_fip(im, segmask, weight_map):
    return im[:,::-1,:], segmask[:,::-1], weight_map[:,::-1]
    
def random_cropping_and_rot(im, segmask, weight_map):
    l, r, t, b = random.uniform(0, 0.12), random.uniform(0, 0.12), random.uniform(0, 0.12), random.uniform(0, 0.12)
    l = int(img_size[0]*l)
    r = img_size[0] - int(img_size[1]*r)
    t = int(img_size[1]*t) 
    b = img_size[1] - int(img_size[0]*b)

    return [
        cv2.resize(im[t:b, l:r], img_size),
        cv2.resize(segmask[t:b, l:r], img_size),
        cv2.resize(weight_map[t:b, l:r], img_size)
        ]

def brightness(img, low=0.6, high=1.3):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def color_augmentation(im):
    if random.random() < 0.5: #Channel shift
        offset = random.uniform(-35, 35)
        im = np.clip(im + offset, 0., 255.)
        im = np.array(im, dtype=np.uint8)
    else:
        im = brightness(im)

    return im


if __name__ == "__main__":
    input_img_paths, target_img_paths, weight_map_paths = get_dataset_paths(SEED1)
    print(f'Total images: {len(input_img_paths)}')

    input_img_paths = input_img_paths[:N]
    target_img_paths = target_img_paths[:N]
    weight_map_paths = weight_map_paths[:N]

    train_x_paths = input_img_paths[:train_samples]
    train_y_paths = target_img_paths[:train_samples]
    train_wm_paths = weight_map_paths[:train_samples]

    random.Random(1).shuffle(train_x_paths)
    random.Random(1).shuffle(train_y_paths)
    random.Random(1).shuffle(train_wm_paths)

    for xp, yp, wmp in zip(train_x_paths, train_y_paths, train_wm_paths):
        if random.random() < 0.35:

            impath = xp.split('/')[-1].replace("\\","_")            

            im = cv2.resize(cv2.imread(xp,cv2.IMREAD_COLOR),img_size)
            sm = cv2.resize(cv2.imread(yp, cv2.IMREAD_GRAYSCALE),img_size)
            wm = np.load(wmp)

            #plt.subplots(1,3)
            #plt.subplot(131)
            #plt.imshow(im[:,:,::-1])
            #plt.subplot(132)
            #plt.imshow(segmask, cmap='gray')
            #plt.subplot(133)
            #plt.imshow(wm, cmap='jet')
            #plt.show()

            r = random.random()
            if r < 0.65:
                im, sm, wm = horizontal_fip(im, sm, wm)
            if r > 0.45:
                im, sm, wm = random_cropping_and_rot(im, sm, wm)
            if random.random() < 0.4:
                im = color_augmentation(im)

            #plt.subplots(1,3)
            #plt.subplot(131)
            #plt.imshow(im[:,:,::-1])
            #plt.subplot(132)
            #plt.imshow(sm, cmap='gray')
            #plt.subplot(133)
            #plt.imshow(wm, cmap='jet')
            #plt.show()

            cv2.imwrite(f"{storepath}{x_store_path}augmented_{impath}", im)
            cv2.imwrite(f"{storepath}{y_store_path}augmented_{impath}", sm)
            np.save(f"{storepath}{wm_store_path}augmented_{impath[:-4]}", wm)

    print("done")

           