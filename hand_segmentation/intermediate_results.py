import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import cv2
import matplotlib.pyplot as plt

from read_images import get_dataset_paths
from train import SEED1, SEED2, N, IMG_SIZE
from egohands_class import egohands
from tools_ import get_IOU, display_results

model_name = "modelV2_PReLU_vlr_cos_aug_FOCAL_o5_1o1_03"

def display_intermediate_results(input_img_path, intermediate_output, layer_name):

    fig, axes = plt.subplots(5,5)
    axes = axes.reshape(-1)

    axes[0].axis('Off')
    axes[0].imshow(cv2.resize(cv2.imread(input_img_path, cv2.IMREAD_COLOR), IMG_SIZE)[:,:,::-1])

    for i, ax in enumerate(axes[1:]):
        ax.axis('Off')
        ax.imshow(intermediate_output[:,:,i])

    plt.subplots_adjust(wspace=0.025, hspace=0.05)
    plt.show()

if __name__ == "__main__":
    ###
    input_img_paths, target_img_paths, weight_map_paths = get_dataset_paths(SEED1)

    input_img_paths = input_img_paths[:N]
    target_img_paths = target_img_paths[:N]
    weight_map_paths = weight_map_paths[:N]

    test_samples = 50 #int(N * 0.05) #reserve 10% of the data for testing

    test_x_paths  = input_img_paths[-test_samples:]
    test_y_paths  = target_img_paths[-test_samples:]
    test_wm_paths = weight_map_paths[-test_samples:]

    #for i in range(5):
    #    print(test_x_paths[100*i])

    random.Random(SEED2).shuffle(test_x_paths)
    random.Random(SEED2).shuffle(test_y_paths)
    random.Random(SEED2).shuffle(test_wm_paths)
    ###
    

    with tf.device("cpu:0"):   
        model = tf.keras.models.load_model("./gen/" + model_name)

        layer_names = [layer.name for layer in model.layers]
        layer_name = 'p_re_lu_5'
        intermediate_layer_model = keras.Model(inputs=model.input,
                                               outputs=model.get_layer(layer_name).output)


        test_gen = egohands(1, IMG_SIZE, test_x_paths, test_y_paths, test_wm_paths)
        intermediate_output = intermediate_layer_model.predict(test_gen)

    for i in range(5):
        display_intermediate_results(test_x_paths[i], intermediate_output[i], layer_name)

