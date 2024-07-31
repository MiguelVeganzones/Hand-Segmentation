import tensorflow as tf
import tensorflow.keras as keras
import random
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import ImageOps
import cv2
from sklearn.utils import class_weight
import os
import time
import tensorflow_model_optimization as tfmot 
import tempfile

from IPython.display import Image, display
#from tensorflow.keras.preprocessing.image import load_img

from egohands_class import egohands
from segmentation_model import get_model, NUM_CLASSES, get_model_with_skip_connections_and_residuals, get_model_with_skip_connections_UNET1, get_model_with_skip_connections_UNET0, get_PReLU_model, get_PReLU_model2
from tools_ import get_IOU, get_pixel_precision, display_results, get_precission, get_recall, get_custom_metric
import sys
sys.path.append(r'./dataset/')
from dataset import get_full_dataset_paths
from directories import INPUT_IMG_DIR, GROUND_TRUTH_DIR, WEIGHT_MAPS_DIR, AUG_INPUT_IMG_DIR, AUG_GROUND_TRUTH_DIR, AUG_WEIGHT_MAPS_DIR
from directories import INPUT_IMG_DIR_640_360, GROUND_TRUTH_DIR_640_360, WEIGHT_MAPS_DIR_640_360, AUG_INPUT_IMG_DIR_640_360, AUG_GROUND_TRUTH_DIR_640_360, AUG_WEIGHT_MAPS_DIR_640_360
from dataset_config import IMG_SIZE as img_size

gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.function(jit_compile = True) 


BATCH_SIZE = 8
SEED2 = 3879 #random.randint(0,10000) # to shuffle training dataset
Train = True
Sparsify = False
CHECKPOINT_EPOCH_FREC = 5

#model_name = "modelV3_PReLU_vlr_cos_aug_FOCAL_o5_1o1_02_640_360"
model_name = "separable_PReLU_640_368"

EPOCHS = 30

#def display_mask(i) -> None:
#    mask = np.argmax(val_preds[i], axis=-1)
#    mask = np.expand_dims(mask, axis=-1)
#    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
#    display(img)

if __name__ == "__main__":

    dataset_paths = get_full_dataset_paths(INPUT_IMG_DIR_640_360, GROUND_TRUTH_DIR_640_360, WEIGHT_MAPS_DIR_640_360, AUG_INPUT_IMG_DIR_640_360, AUG_GROUND_TRUTH_DIR_640_360, AUG_WEIGHT_MAPS_DIR_640_360)

    ############### Train paths ######################

    train_x_paths = [p for dataset in ['train_dataset', 'augmented_dataset'] for video_path in dataset_paths[dataset]['input'] for p in video_path]
    train_y_paths = [p for dataset in ['train_dataset', 'augmented_dataset'] for video_path in dataset_paths[dataset]['ground_truth'] for p in video_path]
    train_wm_paths = [p for dataset in ['train_dataset', 'augmented_dataset'] for video_path in dataset_paths[dataset]['weight_maps'] for p in video_path]

    for i in range(len(train_x_paths) // 100):
        print(train_x_paths[i*100])
    print("################################################")

    random.Random(SEED2).shuffle(train_x_paths)
    random.Random(SEED2).shuffle(train_y_paths)
    random.Random(SEED2).shuffle(train_wm_paths)
    ##################

    val_x_paths = [p for video_path in dataset_paths['validation_dataset']['input'] for p in video_path]
    val_y_paths = [p for video_path in dataset_paths['validation_dataset']['ground_truth'] for p in video_path]
    val_wm_paths = [p for video_path in dataset_paths['validation_dataset']['weight_maps'] for p in video_path]

    test_x_paths = [p for video_path in dataset_paths['test_dataset']['input'] for p in video_path]
    test_y_paths = [p for video_path in dataset_paths['test_dataset']['ground_truth'] for p in video_path]
    test_wm_paths = [p for video_path in dataset_paths['test_dataset']['weight_maps'] for p in video_path]
    
    for i in range(len(test_x_paths) // 100):
        print(test_x_paths[100*i])
    print("################################################")

    for i in range(len(val_x_paths) // 100):
        print(val_x_paths[100*i])
    print("################################################")

    random.Random(SEED2).shuffle(test_x_paths)
    random.Random(SEED2).shuffle(test_y_paths)
    random.Random(SEED2).shuffle(test_wm_paths)

    for path in test_x_paths:
        assert(path not in train_x_paths)
        assert(path not in val_x_paths)

    print(f'Training samples: {len(train_x_paths)}')
    print(f'Validation samples: {len(val_x_paths)}')
    print(f'Test samples: {len(test_x_paths)}')
    print(f"Epoch size: {int(len(train_x_paths)/BATCH_SIZE)}")

    assert(len(train_x_paths) == len(train_y_paths) == len(train_wm_paths))
    assert(len(val_x_paths) == len(val_y_paths) == len(val_wm_paths))
    assert(len(test_x_paths) == len(test_y_paths) == len(test_wm_paths))
    
    train_gen = egohands(BATCH_SIZE, img_size, train_x_paths, train_y_paths, train_wm_paths)
    val_gen = egohands(BATCH_SIZE, img_size, val_x_paths, val_y_paths, val_wm_paths)
    if Train:

        path = rf"./gen/{model_name}"
        if not os.path.exists(path):
            os.makedirs(path)
            os.mkdir(rf"./gen/{model_name}/checkpoints")
        
        #for i in range(25):
        #    train_gen.show(i**2)
            #val_gen.show(2*i)

        model = get_PReLU_model2(img_size[::-1], NUM_CLASSES)
        
        callbacks = [
                keras.callbacks.ModelCheckpoint(
                    #"./gen/" + model_name + "model_check_point.h5",
                    filepath = "./gen/" + model_name + '/checkpoints/{epoch:02d}.hdf5',
                    verbose=1, 
                    save_weights_only=False,
                    save_best_only = False, # TODO Try True
                    save_freq = int( len(train_x_paths) / BATCH_SIZE * CHECKPOINT_EPOCH_FREC )
                    #save_best_only=True)
                    #keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 10, mode = 'max', verbose=1
                ),
        ]

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        t0 = time.time()

        history = model.fit(train_gen, 
                  epochs=EPOCHS, 
                  validation_data=val_gen, 
                  batch_size = BATCH_SIZE,
                  callbacks=callbacks
        )

        t1 = time.time()

        tf.keras.models.save_model(model, f"./gen/{model_name}", include_optimizer=False)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=f"./gen/{model_name}/model.png")

        print(history.history.keys())
        # summarize history for accuracy
        np.save(f"./gen/{model_name}/binary_io_u.npy", history.history['binary_io_u'])
        np.save(f"./gen/{model_name}/val_binary_io_u.npy", history.history['val_binary_io_u'])
        np.save(f"./gen/{model_name}/val_loss.npy", history.history['val_loss'])
        
        plt.plot(history.history['binary_io_u'])
        plt.plot(history.history['val_binary_io_u'])
        plt.title('model IoU')
        plt.ylabel('IoU')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f"{os.getcwd()}/gen/{model_name}/IoU.png")
        plt.savefig(f"{os.getcwd()}/gen/{model_name}/IoU.eps")
        plt.show()
        ## summarize history for mean_io_u
        #plt.plot(history.history['mean_io_u'])
        #plt.plot(history.history['val_mean_io_u'])
        #plt.title('model mean iou')
        #plt.ylabel('mean iou')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        
        print(f"\n\nTRAINING TIME: {(t1-t0)/3600}\n")

    else: 
        model = tf.keras.models.load_model("./gen/" + model_name)

    if Sparsify:
        
        pruning_params = {
            'pruning_schedule' : tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.5,
                                                                      final_sparsity=0.8,
                                                                      begin_step=0,
                                                                      end_step=639*5)
        }

        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        model_for_pruning.summary()
        logdir = tempfile.mkdtemp()
        callbacks = [
                keras.callbacks.ModelCheckpoint(
                    #"./gen/" + model_name + "model_check_point.h5",
                    filepath = "./gen/" + model_name + '/checkpoints/{epoch:02d}.hdf5',
                    verbose=1, 
                    save_weights_only=False,
                    save_best_only = False, # TODO Try True
                    save_freq = int( len(train_x_paths) / BATCH_SIZE * CHECKPOINT_EPOCH_FREC )
                    #save_best_only=True)
                    #keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 10, mode = 'max', verbose=1
                ),
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate = 0.002,
                    decay_steps = EPOCHS * 639,
                    alpha = 0.2
                    )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model_for_pruning.compile(optimizer, 
                  loss = 'binary_crossentropy',
                  metrics = [
                             tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
                            ]
        )

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        history = model_for_pruning.fit(train_gen, 
                  epochs=EPOCHS, 
                  validation_data=val_gen, 
                  batch_size = BATCH_SIZE,
                  callbacks=callbacks
        )
        
        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning) 
        model_for_export.save(rf"./gen/{model_name}_pruned", include_optimizer=False)

    test_gen = egohands(1, img_size, test_x_paths, test_y_paths, test_wm_paths)
    test_preds = model.predict(test_gen)

    #plt.figure()
    #plt.imshow(val_preds[0])
    #plt.show()

    for i in range(25):
        display_results(test_x_paths[i], test_y_paths[i], test_preds[i])

    count = 0
    total_IOU = 0
    total_pixel_precision = 0
    total_precision = 0
    total_recall = 0
    total_cbm = 0

    for target_img_path, inferred in zip(test_y_paths, test_preds):
        test_y = np.expand_dims(np.array(cv2.resize(cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE), img_size) > 128, dtype=np.uint8),2) 
                                                                                        
        #pred = np.array(np.argmax(inferred, axis=-1),dtype = np.uint8)

        pred = inferred > 0.5
        count += 1
        total_IOU += get_IOU(test_y, pred)
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
    print('done')