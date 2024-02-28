import random
from train import SEED1, SEED2, N, Train, test_samples, val_samples, train_samples
from tools_ import get_dataset_paths, get_IOU, get_pixel_precision, display_results, get_precission, get_recall, get_custom_metric
import tensorflow as tf
from egohands_class import egohands
from read_images import img_size, get_augmented_data
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataset.dataset_config import IMG_SIZE

model_name = "modelV2_PReLU_vlr_cos_aug_FOCAL_o5_1o1_03"
model_name2 = "model0_BCE_01"
model_name3 = "model0_BCE_02"

def compare_nets(net1, net2, x_paths, y_paths, wm_paths):
    with tf.device("cpu:0"):   
        model1 = tf.keras.models.load_model("./gen/" + net1)
        model2 = tf.keras.models.load_model("./gen/" + net2)

    test_gen = egohands(1, img_size, x_paths, y_paths, wm_paths)
    test_preds1 = model1.predict(test_gen)
    test_preds2 = model2.predict(test_gen)

    for im_path, target_img_path, inferred1, inferred2 in zip(x_paths, y_paths, test_preds1, test_preds2): 
        fig, axes = plt.subplots(2,3)

        plt.subplot(231)
        plt.imshow(cv2.resize(cv2.imread(im_path, cv2.IMREAD_COLOR), img_size)[:,:,::-1])

        plt.subplot(234)
        train_y = np.expand_dims(np.array(cv2.resize(cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE), img_size) > 128,dtype=np.uint8),2)#// 255, dtype=np.uint8) # > 128,
                                                                                                                                                                                                               
        plt.imshow(train_y, cmap = 'gray')
    
        plt.subplot(232)
        #pred = np.array(np.argmax(inferred, axis=2),dtype = np.uint8)
        pred1 = np.array(inferred1 > 0.5, dtype = np.uint8)
        plt.imshow(pred1, cmap = 'gray')

        plt.subplot(235)
        plt.imshow(np.array(abs(np.array(pred1, dtype = np.short) - np.array(train_y, dtype=np.short)),dtype=np.uint8), cmap = 'hot')

        plt.subplot(233)
        #pred = np.array(np.argmax(inferred, axis=2),dtype = np.uint8)
        pred2 = np.array(inferred2 > 0.5, dtype = np.uint8)
        plt.imshow(pred2, cmap = 'gray')

        plt.subplot(236)
        plt.imshow(np.array(abs(np.array(pred2, dtype = np.short) - np.array(train_y, dtype=np.short)),dtype=np.uint8), cmap = 'hot')

        titles = ['Input img', 'ResNet Prediction', 'U-Net Prediction', 
                  'Ground truth', 'ResNet Error', 'U-Net Error']

        for i, row in enumerate(axes):
            for j ,ax in enumerate(row):
                ax.set_title(titles[i*3+j])
                ax.axis(False)

        fig.tight_layout()
        plt.show()


def display_segresults(input_img_path, target_img_path, inferred):
    x = cv2.resize(cv2.imread(input_img_path,cv2.IMREAD_COLOR), img_size)[:,:,::-1]
    y = np.expand_dims(cv2.resize(cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE), img_size), -1) > 128

    inferred = np.reshape(inferred, IMG_SIZE[::-1])>0.5
    y = np.reshape(y, IMG_SIZE[::-1])
    
    x_overlay = np.array(x, dtype=np.uint16)
    for i in range(3):
        x_overlay[:,:,i][y]//=2
    x_overlay[:,:,0][y] += 100  
    x_overlay[:,:,2][y] += 100  

    y_overlay = np.array(x, dtype=np.uint16)
    for i in range(3):
        y_overlay[:,:,i][inferred]//=2
    y_overlay[:,:,2][inferred] += 150
    y_overlay[:,:,1][inferred] += 150 

    x_overlay = np.clip(x_overlay, 0, 255).astype(np.uint8)
    y_overlay = np.clip(y_overlay, 0, 255).astype(np.uint8)

    plt.figure()
    plt.subplot(121)
    plt.imshow(x_overlay)
    plt.axis('Off')
    plt.subplot(122)
    plt.imshow(y_overlay)
    plt.axis('Off')
    plt.show()
    


def get_IOU_plots():
    train_path = f"./gen/{model_name}/binary_io_u.npy"
    val_path = f"./gen/{model_name}/val_binary_io_u.npy"
    train_path2 = f"./gen/{model_name2}/binary_io_u.npy"
    val_path2 = f"./gen/{model_name2}/val_binary_io_u.npy"
    #train_path3 = f"./gen/{model_name3}/binary_io_u.npy"
    #val_path3 = f"./gen/{model_name3}/val_binary_io_u.npy"

    train_iou = np.load(train_path)
    val_iou = np.load(val_path)
    train_iou2 = np.load(train_path2)
    val_iou2 = np.load(val_path2)
    #train_iou3 = np.load(train_path3)
    #val_iou3 = np.load(val_path3)

    plt.plot(train_iou, 'b-')
    plt.plot(val_iou, 'b--')
    plt.plot(train_iou2, 'r-')
    plt.plot(val_iou2, 'r--')
    #plt.plot(train_iou3, 'g-')
    #plt.plot(val_iou3, 'g--')
    plt.legend(['train dr=0.0', 'validation dr=0.0',
                'train dr=0.1', 'validation dr=0.1'], loc='upper left')
                #'train dr=0.2', 'validation dr=0.2'], loc='upper left')

    plt.title('model IoU')
    plt.ylabel('IoU')
    plt.xlabel('epoch')

    plt.show()


def get_8_plots():
    model = "model0_BCE_0"
    name_arr = [f"{model}{i}" for i in range(8)]
    dr_arr = [f"dr=0{i}" for i in range(8)]

    train_iou_arr = [np.load(f"./gen/{name}/binary_io_u.npy") for name in name_arr]
    for iou in train_iou_arr:
        plt.plot(iou)

    plt.legend(dr_arr)

    plt.title('model IoU')
    plt.ylabel('IoU')
    plt.xlabel('epoch')
    plt.show()

if __name__ == "__main__":
    ###

    input_img_paths, target_img_paths, weight_map_paths = get_dataset_paths(SEED1)
    aug_input_img_paths, aug_target_img_paths, aug_weight_map_paths = get_augmented_data()

    #for i in range(36):
    #    print(input_img_paths[:train_samples][100*i])
    #print("################################################")

    train_x_paths = input_img_paths[:train_samples] + aug_input_img_paths
    train_y_paths = target_img_paths[:train_samples] + aug_target_img_paths
    train_wm_paths = weight_map_paths[:train_samples] + aug_weight_map_paths

    input_img_paths = input_img_paths[:N]
    target_img_paths = target_img_paths[:N]
    weight_map_paths = weight_map_paths[:N]

    val_x_paths  = input_img_paths[train_samples:train_samples+val_samples]
    val_y_paths  = target_img_paths[train_samples:train_samples+val_samples]
    val_wm_paths = weight_map_paths[train_samples:train_samples+val_samples]

    test_x_paths  = input_img_paths[-test_samples:] + val_x_paths
    test_y_paths  = target_img_paths[-test_samples:] + val_y_paths
    test_wm_paths = weight_map_paths[-test_samples:] + val_wm_paths

    #for i in range(5):
    #    print(test_x_paths[100*i])

    #no hace falta pero por que no
    random.Random(SEED2).shuffle(test_x_paths)
    random.Random(SEED2).shuffle(test_y_paths)
    random.Random(SEED2).shuffle(test_wm_paths)
    ###
    for path in test_x_paths:
        assert(path not in train_x_paths)



    #compare_nets("model0_BCE_00", "UNET0_BCE_00", test_x_paths[:20], test_y_paths[:20], test_wm_paths[:20])

    with tf.device("cpu:0"):   
        model = tf.keras.models.load_model("./gen/" + model_name)

    count = 0
    total_IOU = 0
    total_pixel_precision = 0
    total_precision = 0
    total_recall = 0
    total_cbm = 0
    for i in range(12):
        x_paths = test_x_paths[i*100:(i+1)*100]
        y_paths = test_y_paths[i*100:(i+1)*100]
        wm_paths = test_wm_paths[i*100:(i+1)*100]

        test_gen = egohands(1, img_size, x_paths, y_paths, wm_paths)
        test_preds = model.predict(test_gen)

        #for i in range(25):
        #    display_results(x_paths[i], y_paths[i], test_preds[i])

        for input_img_path, target_img_path, inferred in zip(x_paths, y_paths, test_preds):
            test_y = np.expand_dims(np.array(cv2.resize(cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE), img_size) > 128, dtype=np.uint8),2) 
                                                                                        
            #pred = np.array(np.argmax(inferred, axis=-1),dtype = np.uint8)

            pred = inferred > 0.5
            count += 1
            IoU = get_IOU(test_y, pred)
            if IoU < 0.9 or True:
                display_segresults(input_img_path, target_img_path, inferred)
            total_IOU += IoU
            total_pixel_precision += get_pixel_precision(test_y, pred)
            total_precision += get_precission(test_y, pred)
            total_recall += get_recall(test_y, pred)
            total_cbm += get_custom_metric(test_y, pred)

    print(f"IOU average: {total_IOU/count}")
    print(f"Pixel precision avg: {total_pixel_precision/count}")
    print(f"Precision avg: {total_precision/count}")
    print(f"Recall avg: {total_recall/count}")
    print(f"cbm avg: {total_cbm/count}")
    print(f"Count: {count}")