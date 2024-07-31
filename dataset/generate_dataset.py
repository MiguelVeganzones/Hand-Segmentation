import os
import subprocess
import cv2
import time

from download_dataset import download_and_extract_dataset
from weight_maps import compute_and_store_weight_maps
from data_augmentation import data_augmentation
from dataset import get_base_dataset_paths
from dataset_config import ORIGINAL_IMG_SIZE, IMG_SIZE, BINARIZE_THRESHOLD

from directories import ANNOTATIONS_DIR, CURRENT_DIR_PATH
from directories import INPUT_IMG_DIR, INPUT_IMG_DIR_640_360, GROUND_TRUTH_DIR, GROUND_TRUTH_DIR_640_360, WEIGHT_MAPS_DIR, WEIGHT_MAPS_DIR_640_360
from directories import AUG_INPUT_IMG_DIR, AUG_GROUND_TRUTH_DIR, AUG_WEIGHT_MAPS_DIR, AUG_INPUT_IMG_DIR_640_360, AUG_GROUND_TRUTH_DIR_640_360, AUG_WEIGHT_MAPS_DIR_640_360

def process_images():
    if not os.path.exists(ANNOTATIONS_DIR):
        os.mkdir(ANNOTATIONS_DIR)
    if "_GROUND_TRUTH_DISJUNCT_HANDS" not in os.listdir(ANNOTATIONS_DIR):
        subprocess.run([r"C:\Program Files\MATLAB\R2023b\bin\matlab", "-r", rf"run('{CURRENT_DIR_PATH}\process_images.m') ; exit"])

def resize_images(dataset_info, img_size):
    ## Resize input image
    for in_dir, out_dir, key in dataset_info:
            paths = [item for _, d in get_base_dataset_paths().items() 
                    for item in d[key]]
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            for video in paths:
                if in_dir not in video[0]:
                    print(in_dir, out_dir)
                    continue
                dir = os.path.dirname(video[0]).replace(in_dir, out_dir)
                if os.path.exists(dir):
                    continue
                os.makedirs(dir)
                for path in video:
                    resized_path = path.replace(in_dir, out_dir)
                    im =  cv2.resize(cv2.imread(path), img_size)
                    if "GROUND_TRUTH" in path:
                        _, im = cv2.threshold(im, round(BINARIZE_THRESHOLD * 255), 255, cv2.THRESH_BINARY)        
                    cv2.imwrite(resized_path, im)
            

if __name__ == "__main__":
    download_and_extract_dataset()
    print("dataset downloaded")
    process_images()
    while not os.path.exists(GROUND_TRUTH_DIR):
        time.sleep(10)
    while len(os.listdir(GROUND_TRUTH_DIR)) != 48:
        time.sleep(10)
    time.sleep(5)
    print("ground truth generated")
    resize_images([
        [INPUT_IMG_DIR, INPUT_IMG_DIR_640_360, 'input'], 
        [GROUND_TRUTH_DIR, GROUND_TRUTH_DIR_640_360, 'ground_truth']
    ], IMG_SIZE)
    print("images resized")
    compute_and_store_weight_maps([
        [GROUND_TRUTH_DIR, WEIGHT_MAPS_DIR],
        [GROUND_TRUTH_DIR_640_360, WEIGHT_MAPS_DIR_640_360],
    ])
    print("weight maps generated")
    data_augmentation([
        [INPUT_IMG_DIR, AUG_INPUT_IMG_DIR, 
         GROUND_TRUTH_DIR, AUG_GROUND_TRUTH_DIR,
         WEIGHT_MAPS_DIR, AUG_WEIGHT_MAPS_DIR,
         ORIGINAL_IMG_SIZE],
        [INPUT_IMG_DIR_640_360, AUG_INPUT_IMG_DIR_640_360,
         GROUND_TRUTH_DIR_640_360, AUG_GROUND_TRUTH_DIR_640_360,
         WEIGHT_MAPS_DIR_640_360, AUG_WEIGHT_MAPS_DIR_640_360,
         IMG_SIZE]
    ])
    print("data augmentation done")