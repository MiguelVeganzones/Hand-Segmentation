import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION) 
import tf2onnx
import onnx
import numpy as np
import tqdm

from egohands_class import egohands
import time
import os
import sys
sys.path.append(r'./dataset/')
from dataset import get_full_dataset_paths
from directories import INPUT_IMG_DIR, GROUND_TRUTH_DIR, WEIGHT_MAPS_DIR, AUG_INPUT_IMG_DIR, AUG_GROUND_TRUTH_DIR, AUG_WEIGHT_MAPS_DIR
from directories import INPUT_IMG_DIR_640_360, GROUND_TRUTH_DIR_640_360, WEIGHT_MAPS_DIR_640_360, AUG_INPUT_IMG_DIR_640_360, AUG_GROUND_TRUTH_DIR_640_360, AUG_WEIGHT_MAPS_DIR_640_360
from dataset_config import IMG_SIZE as img_size


# TODO 
## https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html

# 1. try TF_ENABLE_ONEDNN_OPTS=1
# 2. Tensorflow options
#   2.1 Use freeze_graph (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)
#   2.2 Use optimize_for_inference (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py)


class iterable():
    def __init__(self, r):
        self.data = r
        self.len = len(r)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[idx]
    

if __name__ == '__main__':

    dataset_paths = get_full_dataset_paths(INPUT_IMG_DIR_640_360, GROUND_TRUTH_DIR_640_360, WEIGHT_MAPS_DIR_640_360, AUG_INPUT_IMG_DIR_640_360, AUG_GROUND_TRUTH_DIR_640_360, AUG_WEIGHT_MAPS_DIR_640_360)
    test_x_paths = [p for video_path in dataset_paths['validation_dataset']['input'] for p in video_path]
    test_y_paths = [p for video_path in dataset_paths['validation_dataset']['ground_truth'] for p in video_path]
    test_wm_paths = [p for video_path in dataset_paths['validation_dataset']['weight_maps'] for p in video_path]
    
    test_gen = egohands(1, img_size, test_x_paths, test_y_paths, test_wm_paths)

    #input_graph_name = 'saved_model.pb'
    #output_graph_name = 'output_graph.pb'
    #input_graph_path = rf"{path}/{input_graph_name}"
    #output_graph_path = rf'{path}/{output_graph_name}'
    #input_saver_def_path = ""
    #input_binary = False
    #output_node_names = "output_node"
    #restore_op_name = "save/restore_all"
    #filename_tensor_name = "save/Const:0"
    #clear_devices = True

#    freeze_graph.freeze_graph(
#        input_graph_path,
#        input_saver_def_path,
#        input_binary,
#        rf'{path}/checkpoints',
#        output_node_names,
#        restore_op_name,
#        filename_tensor_name,
#        output_graph_path,
#        clear_devices,
#        "",
#        "",
#        "",
#        checkpoint_version=saver_pb2.SaverDef.V2) 
#
#    iofdhoehf
#
    #with tf.Session(graph=tf.Graph()) as sess:
        #saver = tf.train.import_meta_graph()


    #full_model = tf.function(lambda inputs: model(inputs))
    #full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])
    #frozen_func = convert_variables_to_constants_v2(full_model)
    #frozen_func.graph.as_graph_def()
    #tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=f"{path}/frozen_models", name="frozen_model.pb", as_text=False)

    #conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.FP16)
    
    #converter = trt.TrtGraphConverterV2(input_saved_model_dir=rf"{path}", conversion_params=conversion_params)
    #converter.convert()
    #converter.save(rf"{path}/trt_model")    

    path = rf'./gen/poor_640_368/'
    # path = rf'./gen/separable'
    
    
    
    with tf.device("/cpu:0"):
        tf.compat.v1.disable_eager_execution()
        tf.config.optimizer.set_jit(True)

        model = tf.keras.models.load_model(path)

        t0 = time.time()
        test_preds = model.predict(test_gen)
        t1 = time.time()
    
        seconds_per_image = (t1-t0)/len(test_x_paths)
        print(f"\n\nInference TIME: {(t1-t0)} s\n")
        print(f"{seconds_per_image*1000} ms/img\n")
        print(f"{1/seconds_per_image} FPS") 
    