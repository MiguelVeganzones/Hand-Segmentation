from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as k
import numpy as np
import cv2
import os

from directories import GROUND_TRUTH_DIR, GROUND_TRUTH_DIR_640_360, AUG_GROUND_TRUTH_DIR, AUG_GROUND_TRUTH_DIR_640_360
from directories import WEIGHT_MAPS_DIR, WEIGHT_MAPS_DIR_640_360, AUG_WEIGHT_MAPS_DIR, AUG_WEIGHT_MAPS_DIR_640_360

"""

w(x) = wc(x) + w0(x)*exp( (-(d1(x)+d2(x))^2) / (2*sigma^2) ) 

"""

def get_weight_map(segmentation: np.array, w0: float=12., sigma: float=5.) -> np.array:
    """
    Returns a per pixel weighted segmentation of the input binarized segmentation
    Used to force the Unet to learn boundaries between separate objects
    Returns an approximation to redice compute time
    """
    if len(np.shape(segmentation)) == 3:
        segmentation = np.reshape(segmentation, np.shape(segmentation)[:-1])

    shape = np.shape(segmentation)

    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    base_background_weight = 0.95
    w =  np.array(segmentation==0, dtype=np.uint8) * base_background_weight + \
        np.array(segmentation!=0, dtype=np.uint8) #*(1-count/size)

    base_img_diagonal = (360 ** 2 + 640 ** 2) ** 0.5 # This algorithm was tuned for 360x640 images
    img_diagonal = (np.shape(segmentation)[0] ** 2 + np.shape(segmentation)[1] ** 2) ** 0.5
    distance_scale_factor = base_img_diagonal / img_diagonal
    
    #if there is only one object, then there is no separation between objects
    if len(contours) >= 2:
        d1 = np.zeros_like(segmentation) + np.inf
        d2 = np.zeros_like(segmentation) + np.inf

        for j in range(shape[0]):
            for i in range(shape[1]):
                dist = sorted([abs(cv2.pointPolygonTest(cnt, (i, j), measureDist=True)) for cnt in contours])
                d1[j,i] = dist[0] * distance_scale_factor #smallest distance
                d2[j,i] = dist[1] * distance_scale_factor #second smallest distance

        w += w0 * np.exp(-((d1+d2)**2)/(2*sigma**2))
    return w 


"""
https://stackoverflow.com/questions/67615051/implementing-binary-cross-entropy-loss-gives-different-answer-than-tensorflows
"""
def weighted_pixelwise_binary_crossentropy(y_true: np.array, y_pred: np.array, pixel_weights: np.array)->np.array:
    y_pred = np.clip(np.array(y_pred,dtype=float), 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred)
    term_1 = y_true * np.log(y_pred)
    return -(term_0+term_1) * pixel_weights


def compute_and_store_weight_maps(dataset_info):
    for in_dir, out_dir in dataset_info:
        #get input mask path
        ground_truth_folders = sorted(
            [
                os.path.join(in_dir, fname)
                for fname in os.listdir(in_dir)
            ]
        )
        ground_truth_img_paths = sorted(
            [
                [os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(".jpg")]
                for folder in ground_truth_folders
                ]
        )

        weight_map_folders = [folder.replace(in_dir, out_dir) for folder in ground_truth_folders]

        for folder_in, folder_out in zip(ground_truth_img_paths, weight_map_folders):
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)
            else:
                continue
            for path_in in folder_in:
                path_out = path_in.replace(".jpg", ".npy").replace(in_dir, out_dir)
                y = np.expand_dims(cv2.imread(path_in, cv2.IMREAD_GRAYSCALE) > 128, -1).astype(np.uint8) #//255 #binarize array
                np.save(path_out, get_weight_map(y))

if __name__ == "__main__":
    compute_and_store_weight_maps()