import os
import random 

from directories import INPUT_IMG_DIR, GROUND_TRUTH_DIR
from directories import AUG_INPUT_IMG_DIR, AUG_GROUND_TRUTH_DIR, AUG_WEIGHT_MAPS_DIR
from dataset_config import DATASET_SPLIT_SEED
from dataset_config import TRAINING_DATASET_VIDEO_COUNT, TEST_DATASET_VIDEO_COUNT, VALIDATION_DATASET_VIDEO_COUNT 
from dataset_config import NUM_VIDEOS, IMAGES_PER_VIDEO

def get_base_dataset_paths():
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
            os.path.join(GROUND_TRUTH_DIR, fname)
            for fname in os.listdir(GROUND_TRUTH_DIR)
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
    
    #assert dataset shape and size
    assert(len(input_img_paths) == len(target_img_paths))
    for i in range(len(input_img_paths)):
        assert(len(input_img_paths[i]) == len(target_img_paths[i]))

    random.Random(DATASET_SPLIT_SEED).shuffle(input_img_paths)
    random.Random(DATASET_SPLIT_SEED).shuffle(target_img_paths)

    chunk_start = [0, TRAINING_DATASET_VIDEO_COUNT, TRAINING_DATASET_VIDEO_COUNT + TEST_DATASET_VIDEO_COUNT]
    chunk_size = [TRAINING_DATASET_VIDEO_COUNT, TEST_DATASET_VIDEO_COUNT, VALIDATION_DATASET_VIDEO_COUNT] 
    
    split_input_img_paths = [input_img_paths[start:start+size] for start, size in zip(chunk_start, chunk_size)]
    split_ground_truth_img_paths = [target_img_paths[start:start+size] for start, size in zip(chunk_start, chunk_size)]
    
    for i in range(3):
        assert(len(split_input_img_paths[i]) == len(split_ground_truth_img_paths[i]))
    
    for i in range(1,3):
        for path in split_input_img_paths[0]:
            assert(path not in split_input_img_paths[1])
            assert(path not in split_input_img_paths[2])
        
    dataset_dict = {
        'train_dataset' : {
            'input' : split_input_img_paths[0],
            'ground_truth' : split_ground_truth_img_paths[0],
        },
        'test_dataset' : {
            'input' : split_input_img_paths[1],
            'ground_truth' : split_ground_truth_img_paths[1],
        },
        'validation_dataset' : {
            'input' : split_input_img_paths[2],
            'ground_truth' : split_ground_truth_img_paths[2],
        },
    }
    return dataset_dict

def get_full_dataset_paths(input_dir, ground_truth_dir, weight_maps_dir, aug_input_dir=None, aug_ground_truth_dir=None, aug_weight_maps_dir=None):
    #get input image path
    input_img_folders = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
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
            os.path.join(ground_truth_dir, fname)
            for fname in os.listdir(ground_truth_dir)
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
            os.path.join(weight_maps_dir, fname)
            for fname in os.listdir(weight_maps_dir)
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
    
    aug_input_img_paths = None
    if aug_input_dir is not None:
        aug_input_img_folders = sorted(
            [
                os.path.join(aug_input_dir, fname)
                for fname in os.listdir(aug_input_dir)
            ]
        )

        aug_input_img_paths = sorted(
            [
                [os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(".jpg")]
                for folder in aug_input_img_folders
                ]
        )

    aug_ground_truth_paths = None
    if aug_ground_truth_dir is not None:
        aug_ground_truth_folders = sorted(
            [
                os.path.join(aug_ground_truth_dir, fname)
                for fname in os.listdir(aug_ground_truth_dir)
            ]
        )

        aug_ground_truth_paths = sorted(
            [
                [os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(".jpg")]
                for folder in aug_ground_truth_folders
                ]
        )

    aug_weight_map_paths = None
    if aug_weight_maps_dir is not None:
        aug_weight_map_folders = sorted(
            [
                os.path.join(aug_weight_maps_dir, fname)
                for fname in os.listdir(aug_weight_maps_dir)
            ]
        )

        aug_weight_map_paths = sorted(
            [
                [os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(".npy")]
                for folder in aug_weight_map_folders
                ]
        )


    #assert dataset shape and size
    assert(len(input_img_paths) == NUM_VIDEOS)
    assert(len(target_img_paths) == NUM_VIDEOS)
    assert(len(weigh_map_paths) == NUM_VIDEOS)
    for i in range(len(input_img_paths)):
        assert(len(input_img_paths[i]) == IMAGES_PER_VIDEO)
        assert(len(target_img_paths[i]) == IMAGES_PER_VIDEO)
        assert(len(weigh_map_paths[i]) == IMAGES_PER_VIDEO)

    random.Random(DATASET_SPLIT_SEED).shuffle(input_img_paths)
    random.Random(DATASET_SPLIT_SEED).shuffle(target_img_paths)
    random.Random(DATASET_SPLIT_SEED).shuffle(weigh_map_paths)

    chunk_start = [0, TRAINING_DATASET_VIDEO_COUNT, TRAINING_DATASET_VIDEO_COUNT + TEST_DATASET_VIDEO_COUNT]
    chunk_size = [TRAINING_DATASET_VIDEO_COUNT, TEST_DATASET_VIDEO_COUNT, VALIDATION_DATASET_VIDEO_COUNT] 
    
    split_input_img_paths = [input_img_paths[start:start+size] for start, size in zip(chunk_start, chunk_size)]
    split_ground_truth_img_paths = [target_img_paths[start:start+size] for start, size in zip(chunk_start, chunk_size)]
    split_weight_map_paths = [weigh_map_paths[start:start+size] for start, size in zip(chunk_start, chunk_size)]
    
    for i in range(3):
        assert(len(split_input_img_paths[i]) == len(split_ground_truth_img_paths[i]))
        assert(len(split_input_img_paths[i]) == len(split_weight_map_paths[i]))
    
    for i in range(1,3):
        for path in split_input_img_paths[0]:
            assert(path not in split_input_img_paths[1])
            assert(path not in split_input_img_paths[2])
        
    dataset_dict = {
        'train_dataset' : {
            'input' : split_input_img_paths[0],
            'ground_truth' : split_ground_truth_img_paths[0],
            'weight_maps' : split_weight_map_paths[0],
        },
        'test_dataset' : {
            'input' : split_input_img_paths[1],
            'ground_truth' : split_ground_truth_img_paths[1],
            'weight_maps' : split_weight_map_paths[1],
        },
        'validation_dataset' : {
            'input' : split_input_img_paths[2],
            'ground_truth' : split_ground_truth_img_paths[2],
            'weight_maps' : split_weight_map_paths[2],
        },
        'augmented_dataset' : {
            'input' : aug_input_img_paths,
            'ground_truth' : aug_ground_truth_paths,
            'weight_maps' : aug_weight_map_paths
        }
    }
    
    for _, d in dataset_dict.items():
        for _, values in d.items():
            if values is not None:
                for _, values_2 in d.items():
                    assert(len(values) == len(values_2))
            
    return dataset_dict
    
    
def get_training_data_paths(in_dir, gt_dir, wm_dir):
    return get_full_dataset_paths(in_dir, gt_dir, wm_dir)['train_dataset']

def get_test_data_paths():
    return get_full_dataset_paths()['test_dataset']

def get_validation_data_paths():
    return get_full_dataset_paths()['validation_dataset']


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
            os.path.join(AUG_GROUND_TRUTH_DIR, fname)
            for fname in os.listdir(AUG_GROUND_TRUTH_DIR)
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


    