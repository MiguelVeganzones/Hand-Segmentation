import cv2
import os
from tensorflow.python.keras.backend import zeros_like
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.ops.variable_scope import default_variable_creator 
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import backend as k
import random
from math import dist
from itertools import permutations

def show_img_grid(input_img_paths: str, target_img_paths: str, idx: int=0, rows: int=3, cols: int=3) -> None:

    imgs = [[cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR), img_size) for path in input_img_paths[idx:idx+rows*cols]],
            [cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), img_size)/255 for path in target_img_paths[idx:idx+rows*cols]]]

    fig = plt.figure()

    for a in range(rows*cols):
        #x_train
        fig.add_subplot(rows, cols*2, 2*a+1)
        plt.imshow(imgs[0][a][:,:,::-1])
        plt.axis('off')
        #y_train
        fig.add_subplot(rows, cols*2, 2*a+2)
        plt.imshow(imgs[1][a], cmap = 'gray')
        plt.axis('off')

    fig.tight_layout()
    plt.show()

def display_img(src: str, color:bool=0) -> None:    
    plt.figure()
    if color:
        plt.imshow(cv2.resize(cv2.imread(src, cv2.IMREAD_COLOR), img_size)[:,:,::-1], cmap=None)
    else:
        plt.imshow(cv2.resize(cv2.imread(src, cv2.IMREAD_GRAYSCALE), img_size), cmap='gray')

def display_results(input_img_path: str, target_img_path: str, inferred: np.array) -> None:
    fig, axes = plt.subplots(2,2)

    plt.subplot(221)
    plt.imshow(cv2.imread(input_img_path, cv2.IMREAD_COLOR)[:,:,::-1])

    plt.subplot(222)
    train_y = np.expand_dims(np.array(cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE) > 128,dtype=np.uint8),2)#// 255, dtype=np.uint8) # > 128,
                                                                                                                                                                                                                 
    plt.imshow(train_y, cmap = 'gray')
    
    plt.subplot(223)
    #pred = np.array(np.argmax(inferred, axis=2),dtype = np.uint8)
    pred = np.array(inferred > 0.5, dtype = np.uint8)
    plt.imshow(pred, cmap = 'gray')

    plt.subplot(224)
    plt.imshow(np.array(abs(np.array(pred, dtype = np.short) - np.array(train_y, dtype=np.short)),dtype=np.uint8), cmap = 'hot')

    titles = ['Input img', 'Ground truth', 'Prediction', 'error']

    for i, row in enumerate(axes):
        for j ,ax in enumerate(row):
            ax.set_title(titles[i*2+j])
            ax.axis(False)

    fig.tight_layout()
    plt.show()
    #element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #print(weighted_dice_loss_np(train_y, np.expand_dims(cv2.erode(train_y, element),axis=-1), get_weight_map_approx(train_y)))

def get_IOU(target_seg: np.array, inferred_seg: np.array) -> float:
    """
    Inputs two 2D binary numpy arrays 
    """

    a = np.count_nonzero(inferred_seg)
    b = np.count_nonzero(target_seg)
    c = np.count_nonzero(np.logical_or(inferred_seg, target_seg))

    if c == 0:
        if a == 0 and b == 0:
            return 1
        else:
            return 0

    IOU = (a+b-c)/c 
    #print(IOU)

    return IOU   

def get_pixel_precision(target_seg, inferred_seg):
    intersect = inferred_seg == target_seg
    return np.sum(intersect)/np.size(intersect)

def get_precission(target_seg, inferred_seg):
    b = np.count_nonzero(inferred_seg)
    if b:
        return np.count_nonzero(np.logical_and(inferred_seg == target_seg, target_seg))/b
    else:
        return 0

def get_recall(target_seg, inferred_seg):
    b = np.count_nonzero(target_seg)
    if b:
        return np.count_nonzero(np.logical_and(inferred_seg == target_seg, target_seg))/b
    else:
        return 0

def get_custom_metric(target_seg: np.array, inferred_seg: np.array) -> float:
    """
    Inputs target_seg and inferred_seg as binary matrices
    """

    assert(np.shape(target_seg) == np.shape(inferred_seg))

    #target centroids
    target_c = []
    target_seg = np.array(target_seg, dtype=np.uint8)
    contours, _ = cv2.findContours(target_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            target_c.append([cX, cY])

    #target centroids
    pred_c = []
    inferred_seg = np.array(inferred_seg, dtype=np.uint8)
    contours, _ = cv2.findContours(inferred_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            pred_c.append([cX, cY])

    s, l = sorted([target_c, pred_c], key=lambda x : len(x))

    if not s and l:
        return 0.
    elif not l and not s:
        return 1.

    #lines = np.zeros_like(inferred_seg)
    #im = np.dstack((target_seg*255, inferred_seg*255, lines)).astype(np.uint8)
    if len(l) == len(s):
        n = len(l)
        dist_matrix = [[dist(p, q) for p in l] for q in s]
        perms = permutations([i for i in range(n)])
        dist_combinations = [
                    sum([dist_matrix[j][i] for j, i in enumerate(perm)]) 
                    for perm in perms
                ]
        total_dist = min(dist_combinations)

        #for [x1, y1], [x2, y2] in zip(l, s):
        #    line_thickness = 2
        #    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)


    else:
        total_dist = sum([min([dist(p, q) for q in s]) for p in l])
        #d = [min([dist(p, q) for q in s]) for p in l]
        #lines = np.zeros_like(inferred_seg)
        #im = np.dstack((target_seg*255, inferred_seg*255, lines)).astype(np.uint8)
        #for p in l:
        #    for q in s:
        #        if dist(p, q) in d:
        #            cv2.line(im, p, q, (0, 0, 255), thickness=2)
        #plt.imshow(im)
        #plt.axis('Off')
        #plt.show()
    #print(s)
    #print(l)
    #plt.imshow(target_seg, cmap='gray')
    #plt.show()

    diag = (np.shape(target_seg)[0]**2 + np.shape(target_seg)[1]**2)**.5
    #print(sum(dist_arr))
    #print(l, s)
    ret = 1 - (total_dist/len(s)/diag)**.5
    print(ret)

    #plt.subplot(211)
    #plt.imshow(target_seg, cmap='gray')
    #plt.subplot(212)
    #plt.imshow(inferred_seg, cmap='gray')
    #plt.show()

    return ret



def get_weight_map_approx(segmentation: np.array, w0: float=12., sigma: float=5., stride: int=4) -> np.array:
    """
    Returns a per pixel weighted segmentation of the input binarized segmentation
    Used to force the Unet to learn boundaries between separate objects
    Returns an approximation to redice compute time
    """

    shape = np.shape(segmentation)
    
    w = np.array( 
        np.array(segmentation==0, dtype=np.uint8) * 0.95 + \
        np.array(segmentation!=0, dtype=np.uint8) * 1
    ).astype(np.float32)
    #w = np.ones_like(segmentation).astype(np.float32)

    #if there is only one object, then there is no separation between objects

    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) >= 2:
        d1 = np.zeros_like(segmentation) + np.inf
        d2 = np.zeros_like(segmentation) + np.inf
        skip = False 
        j = 0
        #for j in range(shape[0]):
        #    for i in range(shape[1]):
        #        if segmentation[j,i] == 0:
        #            dist = sorted([abs(cv2.pointPolygonTest(cnt, (i, j), measureDist=True)) for cnt in contours])
        #            d1[j,i] = dist[0] #smallest distance
        #            d2[j,i] = dist[1] #second smallest distance

        while j < shape[0]:
            i = 0
            if skip:
                skip = False
                j+=stride
                continue
            skip = True
            while i < shape[1]:
                if (not skip) and (segmentation[j,i] == 0):
                    dist = sorted([abs(cv2.pointPolygonTest(cnt, (i, j), measureDist=True)) for cnt in contours])
                    d1[j,i] = dist[0] #smallest distance
                    d2[j,i] = dist[1] #second smallest distance

                elif segmentation[j,i] == 1:
                    skip = False

                else:
                    i+=stride
                    continue

                i+=1
            j+=1

        w += w0 * np.exp(-((d1+d2)**2)/(2*sigma**2))
    return w 

def get_weight_map(segmentation: np.array, w0: float=12., sigma: float=5) -> np.array:
    """
    Returns a per pixel weighted segmentation of the input binarized segmentation
    Used to force the Unet to learn boundaries between separate objects
    Returns an approximation to redice compute time
    """

    shape = np.shape(segmentation)
    size = np.size(segmentation)
    count = np.count_nonzero(segmentation)

    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

    #w = np.array(segmentation==0, dtype=np.uint8)*count/size + \
    #    np.array(segmentation!=0, dtype=np.uint8)*(1-count/size)
    w = np.array(segmentation==0, dtype=np.uint8) * 0.95 + \
        np.array(segmentation!=0, dtype=np.uint8) * 1

    #if there is only one object, then there is no separation between objects
    if len(contours) >= 2:
        d1 = np.zeros_like(segmentation) + np.inf
        d2 = np.zeros_like(segmentation) + np.inf

        for j in range(shape[0]):
            for i in range(shape[1]):
                if segmentation[j,i] == 0:
                    dist = sorted([abs(cv2.pointPolygonTest(cnt, (i, j), measureDist=True)) for cnt in contours])
                    d1[j,i] = dist[0] #smallest distance
                    d2[j,i] = dist[1] #second smallest distance

        w += w0 * np.exp(-((d1+d2)**2)/(2*sigma**2))
    return w 

def weighted_pixelwise_binary_crossentropy(y_true: np.array, y_pred: np.array, pixel_weights: np.array)->np.array:
    y_pred = np.clip(np.array(y_pred,dtype=float), 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred)
    term_1 = y_true * np.log(y_pred)
    return - (term_0+term_1) * pixel_weights

def weighted_pixelwise_binary_crossentropy_tf(y_true, y_pred, weight_map):    
    y_pred = k.clip(y_pred, k.epsilon(), 1. - k.epsilon())
    term_0 = tf.cast((1-y_true), dtype=tf.float32) * k.log(1.-y_pred + k.epsilon())
    term_1 = tf.cast(y_true, dtype=tf.float32) * k.log(y_pred + k.epsilon())
    return - k.mean((term_0+term_1) * weight_map)

def weighted_dice_loss_tf(y_true, y_pred, weight_map):
    #B = tf.cast(k.greater(y_pred, 0.5), dtype=tf.float32)
    #B = k.clip(y_pred, k.epsilon(), 1.-k.epsilon())
    B = 1./(1. + tf.math.exp(-20*(y_pred - 0.5)))
    A = tf.cast(y_true, dtype = tf.float32)
    A_u_B = tf.reduce_sum(A * B * weight_map)
    A_and_B = tf.math.reduce_sum(A * weight_map) + \
              tf.math.reduce_sum(B * weight_map)
    
    return 1. - 2.*A_u_B/( A_and_B + k.epsilon() )

def weighted_dice_loss_np(y_true, y_pred, weight_map):
    B = np.clip(y_pred, 0., 1.)
    A = np.array(y_true).astype(np.float32)
    intersection = A * B * weight_map
    A_u_B = np.sum(intersection)
    A_and_B = np.sum((A+B) * weight_map)
    
    return 1.- 2.*(A_u_B/A_and_B)

def weighted_pixelwise_focal_loss_tf(y_true, y_pred, weight_map, gamma = 1.1, alpha = 0.5):
    """
    Se ignora el valor del foreground del weightmap. Se ajustan los desequilibrios ente clases con alpha
    """

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred)) #y_pred en el foreground, 1 else
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred)) #y_pred en el background, 0 else
    
    pt_1 = k.clip(pt_1, k.epsilon(), 1. - k.epsilon())
    pt_0 = k.clip(pt_0, k.epsilon(), 1. - k.epsilon())

    weight_map_bg = tf.where(tf.equal(y_true, 0), weight_map, tf.zeros_like(weight_map)) + k.epsilon()
	
    return -k.mean(
               alpha * k.pow(1. - pt_1, gamma) * k.log(pt_1) + 
               (1. - alpha) * k.pow(pt_0, gamma) *  weight_map_bg * k.log(1. - pt_0)
            )


if __name__ == "__main__":

    
    compute_and_store_weight_maps()
    exit()

    _, target_paths, weight_maps = get_dataset_paths(0) 

    segmentation = [np.array(
        cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), img_size)>128,
        dtype=np.uint8) for path in target_paths[:100]]
    

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    for seg, wmap in zip(segmentation[:10], weight_maps[:10]):
        seg_eroded = cv2.dilate(seg, element)
        plt.figure()
        plt.imshow(weighted_dice_loss_np(seg, seg_eroded, wmap), cmap='jet')
        plt.show()

    t0 = time.time()
    for seg in segmentation[:100]:
        get_weight_map(seg)
        plt.figure()
        plt.subplot(121)
        plt.axis('off')
        plt.imshow(seg, cmap = 'gray')
        plt.subplot(122)
        plt.axis('off')
        plt.imshow(get_weight_map(seg), cmap='jet')
        plt.colorbar()
        plt.show()
    t1 = time.time()
    print(t1-t0)
    