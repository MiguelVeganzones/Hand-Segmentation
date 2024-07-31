from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
from tqdm import tqdm 
from dataset_config import BINARIZE_THRESHOLD
from dataset import get_training_data_paths

"""
Random cropping
Small rotations
Horizontal flipping
Saturation and color changes
"""

def horizontal_fip(im, segmask, weight_map):
    return im[:,::-1,:], segmask[:,::-1], weight_map[:,::-1]
    
def random_cropping_and_rot(im, segmask, weight_map, img_size):
    l, r, t, b = random.uniform(0, 0.2), random.uniform(0, 0.2), random.uniform(0, 0.2), random.uniform(0, 0.2)
    l = int(img_size[0]*l)
    r = img_size[0] - int(img_size[1]*r)
    t = int(img_size[1]*t) 
    b = img_size[1] - int(img_size[0]*b)

    _, gt = cv2.threshold(segmask[t:b, l:r], round(BINARIZE_THRESHOLD * 255), 255, cv2.THRESH_BINARY)
    return [
        cv2.resize(im[t:b, l:r], img_size),
        cv2.resize(gt, img_size),
        cv2.resize(weight_map[t:b, l:r], img_size)
        ]

def hue(img, low=0.8, high=1.2):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,0] = hsv[:,:,0]*value 
    hsv = np.clip(hsv, 0., 255.)
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def saturation(img, low=0.5, high=1.4):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value 
    hsv = np.clip(hsv, 0., 255.)
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def brightness(img, low=0.5, high=1.4):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv = np.clip(hsv, 0., 255.)
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def color_augmentation(im):
    if random.random() < 0.4:
        im = hue(im)
    if random.random() < 0.4:
        im  = saturation(im)
    if random.random() < 0.4:
        im = brightness(im)
    if random.random() < 0.15:
        idx = [0, 1, 2] 
        random.shuffle(idx)
        im_copy = np.array(im, copy=True)
        im[:, :, 0] = im_copy[:,:, idx[0]]
        im[:, :, 1] = im_copy[:,:, idx[1]]
        im[:, :, 2] = im_copy[:,:, idx[2]]
    return im

def add_noise(im):
    noise = np.random.normal(loc=0, scale=random.uniform(1, 5), size=np.shape(im))
    im = np.clip(im + noise, 0., 255.)
    im = np.array(im, dtype=np.uint8)
    return im

def data_augmentation(dataset_info):
    for in_dir, aug_in_dir, gt_dir, aug_gt_dir, wm_dir, aug_wm_dir, img_size in tqdm(dataset_info):
        train_dataset = get_training_data_paths(in_dir, gt_dir, wm_dir)
        train_x_paths = train_dataset['input']
        train_y_paths = train_dataset['ground_truth']
        train_wm_paths = train_dataset['weight_maps']

        if not os.path.exists(aug_in_dir):
            os.makedirs(aug_in_dir)
        if not os.path.exists(aug_gt_dir):
            os.makedirs(aug_gt_dir)
        if not os.path.exists(aug_wm_dir):
            os.makedirs(aug_wm_dir)
        for xp_video, yp_video, wmp_video in zip(train_x_paths, train_y_paths, train_wm_paths):
            in_video_path = os.path.dirname(xp_video[0]).replace(in_dir, aug_in_dir)
            gt_video_path = os.path.dirname(yp_video[0]).replace(gt_dir, aug_gt_dir)
            wm_video_path = os.path.dirname(wmp_video[0]).replace(wm_dir, aug_wm_dir)
            if os.path.exists(in_video_path):
                continue
            os.makedirs(in_video_path)
            os.makedirs(gt_video_path)
            os.makedirs(wm_video_path)
            for xp, yp, wmp in zip(xp_video, yp_video, wmp_video):
                for letter in ['a', 'b']:
                    if random.random() < 0.7:
                        im = cv2.imread(xp, cv2.IMREAD_COLOR)
                        sm = cv2.imread(yp, cv2.IMREAD_GRAYSCALE)
                        wm = np.load(wmp)

                        #plt.subplots(1,3)
                        #plt.subplot(131)
                        #plt.imshow(im[:,:,::-1])
                        #plt.subplot(132)
                        #plt.imshow(segmask, cmap='gray')
                        #plt.subplot(133)
                        #plt.imshow(wm, cmap='jet')
                        #plt.show()

                        transformed = false
                        if random.random() < 0.9:
                            im, sm, wm = horizontal_fip(im, sm, wm)
                            transformed = true
                        if random.random() < 0.75:
                            im, sm, wm = random_cropping_and_rot(im, sm, wm, img_size)
                            transformed = true
                        if random.random() < 0.5:
                            im = color_augmentation(im)
                            transformed = true
                        if random.random() < 0.3:
                            im = add_noise(im)
                            transformed = true

                        #plt.subplots(1,3)
                        #plt.subplot(131)
                        #plt.imshow(im[:,:,::-1])
                        #plt.subplot(132)
                        #plt.imshow(sm, cmap='gray')
                        #plt.subplot(133)
                        #plt.imshow(wm, cmap='jet')
                        #plt.show()

                        if not transformed:
                            continue

                        cv2.imwrite(xp.replace(in_dir, aug_in_dir).replace('frame_', f'frame_{letter}_'), im)
                        cv2.imwrite(yp.replace(gt_dir, aug_gt_dir).replace('frame_', f'frame_{letter}_'), sm)
                        np.save(wmp.replace(wm_dir, aug_wm_dir).replace('frame_', f'frame_{letter}_'), wm)

        print("done")
