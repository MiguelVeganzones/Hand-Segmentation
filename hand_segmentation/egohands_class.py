import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img
import cv2
from read_images import img_size
from tools_ import get_weight_map, get_weight_map_approx

class egohands(Sequence):
    """
    Helper to iterate over the data. 
    From: https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """
    def __init__(self, batch_size: int, img_size: tuple[int, int],
                 input_img_paths: list[str], target_img_paths: list[str], 
                 weight_map_paths: list[str]):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.weight_map_paths = weight_map_paths


    def __len__(self):
        """
        return number of images
        """
        return len(self.input_img_paths) // self.batch_size
        #return reduce(lambda count, l: count + len(l), self.input_paths[0], 0)

    def __getitem__(self, idx):
        """
        Return tuple(tuple, tuple) containing images and masks, 
        with a size of self.batch_size
        """
        i = idx * self.batch_size #stride == batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        batch_weight_map_paths = self.weight_map_paths[i : i + self.batch_size]
       
        x = np.zeros((self.batch_size,) + self.img_size[::-1] + (3,), dtype = 'uint8')
        for j, path in enumerate(batch_input_img_paths):
            x[j] = cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR), self.img_size)

        y = np.zeros((self.batch_size,) + self.img_size[::-1] + (1,), dtype = 'uint8')
        #y = np.zeros((self.batch_size,) + self.img_size[::-1] + (1,), dtype = 'uint8')
        for j, path in enumerate(batch_target_img_paths):
            y[j] = np.expand_dims(
                cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), self.img_size) > 128, -1) #//255 #binarize array

        #load weight maps
        w = np.zeros((self.batch_size,) + self.img_size[::-1] + (1,), dtype = 'float32') #weight maps
        for j, path in enumerate(batch_weight_map_paths):
            w[j] = np.reshape(np.load(path), self.img_size[::-1]+(1,))

        return x, y, w

    def show(self, idx: int=0, rows: int=3, cols: int=2) -> None:

        fig = plt.figure()
        for a in range(rows*cols):
            x, y, w = self.__getitem__(idx + a);
            #x_train
            fig.add_subplot(rows, cols*3, 3*a+1)
            plt.imshow(x[0][:,:,::-1])
            plt.axis('off')
            #y_train
            fig.add_subplot(rows, cols*3, 3*a+2)
            plt.imshow(y[0], cmap = 'gray')
            plt.axis('off')
            #weight map
            fig.add_subplot(rows, cols*3, 3*a+3)
            plt.imshow(w[0], cmap='jet')
            plt.axis('off')

        fig.tight_layout()
        plt.show()
        
    #def compute_weights(self): 
    #    w = np.zeros((len(self.target_img_paths),) + self.img_size[::-1] + (1,), dtype = 'float32') #weight maps
    #    #y = np.zeros((self.batch_size,) + self.img_size[::-1] + (1,), dtype = 'uint8')
    #    for j, path in enumerate(self.target_img_paths):
    #        if j%50 == 0:
    #            print(j)
    #        y = np.expand_dims(
    #             cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), self.img_size) > 128, -1).astype(np.uint8) #//255 #binarize array
    #        w[j] = get_weight_map_approx(y)
    #    return w


        
    #    class egohands(Sequence):
    #"""
    #Helper to iterate over the data. 
    #From: https://keras.io/examples/vision/oxford_pets_image_segmentation/
    #"""
    #def __init__(self, batch_size: int, img_size: tuple[int, int],
    #             input_img_paths: list[str], target_img_paths: list[str]):
    #    self.batch_size = batch_size
    #    self.img_size = img_size
    #    self.input_img_paths = input_img_paths
    #    self.target_img_paths = target_img_paths

    #def __len__(self):
    #    """
    #    return number of images
    #    """
    #    return len(self.input_img_paths) // self.batch_size
    #    #return reduce(lambda count, l: count + len(l), self.input_paths[0], 0)

    #def __getitem__(self, idx):
    #    """
    #    Return tuple(tuple, tuple) containing images and masks, 
    #    with a size of self.batch_size
    #    """
    #    i = idx * self.batch_size #stride == batch_size in this case
    #    batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
    #    batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
       
    #    x = np.zeros((self.batch_size,) + self.img_size[::-1] + (3,), dtype = 'uint8')
    #    for j, path in enumerate(batch_input_img_paths):
    #        x[j] = cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR), self.img_size)

    #    y = np.zeros((self.batch_size,) + self.img_size[::-1] + (1,), dtype = 'uint8')
    #    for j, path in enumerate(batch_target_img_paths):
    #         y[j] = np.expand_dims(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), self.img_size), 2) > 128 #//255 #binarize array
    #         #y[j] = np.expand_dims(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), self.img_size), 2)>128 #binarize array
    #        #y[j] = np.array(np.expand_dims(
    #        #    cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), self.img_size)>128, 2), dtype=np.uint8) #binarize array
    #        #plt.figure()
    #        #plt.imshow(y[j], cmap = 'gray')
    #        #plt.show()
    #    return x, y

    #def show(self, idx: int=0, rows: int=3, cols: int=3) -> None:
    #    imgs = [[cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR), self.img_size) for path in self.input_img_paths[idx:idx+rows*cols]],
    #            [cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), self.img_size)/255 for path in self.target_img_paths[idx:idx+rows*cols]]]

    #    fig = plt.figure()
    #    for a in range(rows*cols):
    #        #x_train
    #        fig.add_subplot(rows, cols*2, 2*a+1)
    #        plt.imshow(imgs[0][a][:,:,::-1])
    #        plt.axis('off')
    #        #y_train
    #        fig.add_subplot(rows, cols*2, 2*a+2)
    #        plt.imshow(imgs[1][a], cmap = 'gray')
    #        plt.axis('off')

    #    fig.tight_layout()
    #    plt.show()
        
        