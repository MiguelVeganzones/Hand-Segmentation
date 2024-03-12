import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import random
sys.path.append(r'./dataset/')
from dataset import get_full_dataset_paths
from directories import INPUT_IMG_DIR, GROUND_TRUTH_DIR, WEIGHT_MAPS_DIR, AUG_INPUT_IMG_DIR, AUG_GROUND_TRUTH_DIR, AUG_WEIGHT_MAPS_DIR
from directories import INPUT_IMG_DIR_640_360, GROUND_TRUTH_DIR_640_360, WEIGHT_MAPS_DIR_640_360, AUG_INPUT_IMG_DIR_640_360, AUG_GROUND_TRUTH_DIR_640_360, AUG_WEIGHT_MAPS_DIR_640_360, BALANCED_AUG_WEIGHT_MAPS_DIR_640_360, BALANCED_WEIGHT_MAPS_DIR_640_360

if __name__ == "__main__":
    rows = 2
    cols = 1
    idx = 0

    s = 100*idx #start, as there are 100 img per video, idx refers to each video
    
    dataset_paths = get_full_dataset_paths(INPUT_IMG_DIR_640_360, GROUND_TRUTH_DIR_640_360, BALANCED_WEIGHT_MAPS_DIR_640_360,
                                           AUG_INPUT_IMG_DIR_640_360, AUG_GROUND_TRUTH_DIR_640_360, BALANCED_AUG_WEIGHT_MAPS_DIR_640_360)

    input_img_paths = [p for dataset in ['train_dataset', 'augmented_dataset', 'test_dataset', 'validation_dataset'] for video_path in  dataset_paths[dataset]['input'] for p in video_path]
    target_img_paths = [p for dataset in ['train_dataset', 'augmented_dataset', 'test_dataset', 'validation_dataset'] for video_path in dataset_paths[dataset]['ground_truth'] for p in video_path]
    wm_paths = [p for dataset in ['train_dataset', 'augmented_dataset', 'test_dataset', 'validation_dataset'] for video_path in dataset_paths[dataset]['weight_maps'] for p in video_path]

    seed = 23534
    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)
    random.Random(seed).shuffle(wm_paths) 
    
    imgs = [[cv2.imread(path, cv2.IMREAD_COLOR) for path in input_img_paths[0:100]],
            [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in target_img_paths[0:100]],
            [np.load(path) for path in wm_paths[0:100]]]

    for i in range(0, 100):
        axes = []
        fig = plt.figure()
        for a in range(rows*cols):
            #x
            axes.append(fig.add_subplot(rows, cols*3, 3*a+1))
            plt.imshow(imgs[0][a+i][:,:,::-1])
            plt.axis('off')
            #y
            axes.append(fig.add_subplot(rows, cols*3, 3*a+2))
            plt.imshow(imgs[1][a+i], cmap = 'gray')
            plt.axis('off')
            #wm 
            axes.append(fig.add_subplot(rows, cols*3, 3*a+3))
            plt.imshow(imgs[2][a+i], cmap='jet')
            plt.colorbar()
            plt.axis('off')
        fig.tight_layout() 
        plt.show() 

