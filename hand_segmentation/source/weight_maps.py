from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as k
import numpy as np
import cv2
from read_images import img_size

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
    size = np.size(segmentation)
    count = np.count_nonzero(segmentation)

    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    w =  np.array(segmentation==0, dtype=np.uint8)*count/size + \
        np.array(segmentation!=0, dtype=np.uint8)*(1-count/size)

    #if there is only one object, then there is no separation between objects
    if len(contours) >= 2:
        d1 = np.zeros_like(segmentation) + np.inf
        d2 = np.zeros_like(segmentation) + np.inf

        for j in range(shape[0]):
            for i in range(shape[1]):
                dist = sorted([abs(cv2.pointPolygonTest(cnt, (i, j), measureDist=True)) for cnt in contours])
                d1[j,i] = dist[0] #smallest distance
                d2[j,i] = dist[1] #second smallest distance

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



def get_weight_map(segmentation_tf, w0: float=12., sigma: float=5., stride: int=4):
    """
    Returns a per pixel weighted segmentation of the input binarized segmentation
    Used to force the Unet to learn boundaries between separate objects
    Returns an approximation to redice compute time
    """

    #### model.compile(..., run_eagerly = True) : segmentation_tf.numpy()
    segmentation = np.array(segmentation_tf.numpy(),dtype=np.uint8).reshape(img_size)

    shape = np.shape(segmentation)
    size = np.size(segmentation)
    count = np.count_nonzero(segmentation)

    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1)

    w =  np.array(segmentation==0, dtype=np.uint8)*count/size + \
        np.array(segmentation!=0, dtype=np.uint8)*(1-count/size)

    #if there is only one object, then there is no separation between objects
    if len(contours) >= 2:
        d1 = np.zeros_like(segmentation) + np.inf
        d2 = np.zeros_like(segmentation) + np.inf
        skip = False 
        j = 0
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
        w += + w0 * np.exp(-((d1+d2)**2)/(2*sigma**2))
    return k.constant(w) 

def weighted_pixelwise_binary_crossentropy(y_true, y_pred):
    w = get_weight_map(y_true)
    
    y_pred = k.clip(y_pred, k.epsilon(), 1 - k.epsilon())
    term_0 = tf.cast((1-y_true), dtype=tf.float32) * k.log(1-y_pred + k.epsilon())
    term_1 = tf.cast(y_true, dtype=tf.float32) * k.log(y_pred + k.epsilon())
    
    return -(term_0+term_1) * w