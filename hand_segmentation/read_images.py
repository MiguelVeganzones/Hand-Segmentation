import os
import cv2
from matplotlib import pyplot as plt
import random

INPUT_IMG_DIR = "../../dataset/egohands/_LABELLED_SAMPLES/"
TARGET_IMG_DIR = "../../dataset/annotations/_GROUND_TRUTH_DISJUNCT_HANDS/"
WEIGHT_MAPS_DIR = "../../dataset/annotations/_WEIGHT_MAPS/"

AUG_INPUT_IMG_DIR =   "../../dataset/egohands/_AUGMENTED_SAMPLES/_LABELLED_SAMPLES/"
AUG_TARGET_IMG_DIR =  "../../dataset/annotations/_AUGMENTED_SAMPLES/_GROUND_TRUTH_DISJUNCT_HANDS/"
AUG_WEIGHT_MAPS_DIR = "../../dataset/annotations/_AUGMENTED_SAMPLES/_WEIGHT_MAPS/"

img_size = (1280//2, 640//2) # ()

def get_augmented_data():
    aug_input_img_paths = sorted(
        [
            os.path.join(AUG_INPUT_IMG_DIR, fname)
            for fname in os.listdir(AUG_INPUT_IMG_DIR)
            if fname.endswith(".jpg")
        ]
    )

    aug_target_img_paths = sorted(
        [
            os.path.join(AUG_TARGET_IMG_DIR, fname)
            for fname in os.listdir(AUG_TARGET_IMG_DIR)
            if fname.endswith(".jpg")
        ]
    )

    aug_weight_map_paths = sorted(
        [
            os.path.join(AUG_WEIGHT_MAPS_DIR, fname)
            for fname in os.listdir(AUG_WEIGHT_MAPS_DIR)
            if fname.endswith(".npy")
        ]
    )

    return aug_input_img_paths, aug_target_img_paths, aug_weight_map_paths

def get_dataset_paths(seed: int) -> tuple[list, list, list]:
    """
    returns two lists, each contains a list of .jpg paths per video
    return type: tuple[ list[ path ], list[ path ] ]
    """
    #get input image path
    input_img_folders = sorted(
        [
            os.path.join(INPUT_IMG_DIR, fname)
            for fname in os.listdir(INPUT_IMG_DIR)
        ]
    )

    input_img_paths = sorted(
        [
            [os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.endswith(".jpg")]
            for folder in input_img_folders
            ]
    )

    #get input mask path
    target_img_folders = sorted(
        [
            os.path.join(TARGET_IMG_DIR, fname)
            for fname in os.listdir(TARGET_IMG_DIR)
        ]
    )

    target_img_paths = sorted(
        [
            [os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.endswith(".jpg")]
            for folder in target_img_folders
            ]
    )

    #get weight map paths
    weigh_map_folders = sorted(
        [
            os.path.join(WEIGHT_MAPS_DIR, fname)
            for fname in os.listdir(WEIGHT_MAPS_DIR)
        ]
    )

    weigh_map_paths = sorted(
        [
            [os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.endswith(".npy")]
            for folder in weigh_map_folders
            ]
    )

    #assert dataset shape and size
    assert(len(input_img_paths) == len(target_img_paths) == len(weigh_map_paths))
    for i in range(len(input_img_paths)):
        assert(len(input_img_paths[i]) == len(target_img_paths[i]) == len(weigh_map_paths[i]))

    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)
    random.Random(seed).shuffle(weigh_map_paths)

    return ([item for sublist in input_img_paths for item in sublist],
            [item for sublist in target_img_paths for item in sublist],
            [item for sublist in weigh_map_paths for item in sublist])


##target path
#target_img_paths = sorted(
#    [
#        os.path.join(input_dir, fname)
#        for fname in os.listdir(target_dir)
#        if fname.endswith(".JPEG") and not fname.startswith(".")
#    ])


if __name__ == "__main__":
    rows = 3
    cols = 3
    idx = 0

    s = 100*idx #start, as there are 100 img per video, idx refers to each video

    input_img_paths, target_img_paths = get_dataset_paths()

    imgs = [[cv2.imread(path, cv2.IMREAD_COLOR) for path in input_img_paths[s:s+rows*cols]],
            [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in target_img_paths[s:s+rows*cols]]]

    axes = []
    fig = plt.figure()

    for a in range(rows*cols):
        #x_train
        axes.append(fig.add_subplot(rows, cols*2, 2*a+1))
        plt.imshow(imgs[0][a][:,:,::-1])
        plt.axis('off')
        #y_train
        axes.append(fig.add_subplot(rows, cols*2, 2*a+2))
        plt.imshow(imgs[1][a], cmap = 'gray')
        plt.axis('off')

    fig.tight_layout()
    plt.show()

