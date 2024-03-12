import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
import visualkeras
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.append(r'./dataset/')
from dataset_config import IMG_SIZE as img_size

from tools_ import weighted_pixelwise_binary_crossentropy_tf, weighted_dice_loss_tf, weighted_pixelwise_focal_loss_tf

NUM_CLASSES = 1
model_name = "model_10_03_22"

initial_learning_rate = 0.0017
final_learning_rate = 0.0006
epochs = 30
steps_per_epoch = 951
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)

### Based on https://keras.io/examples/vision/oxford_pets_image_segmentation/

class custom_model(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, weight_map = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = weighted_pixelwise_focal_loss_tf(y, y_pred, weight_map)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def get_model(img_size, num_classes):
    """
    img_size follows numpy's format (y,x)
    """

    initializer = 'he_normal'
    _dropout = 0.3
    inputs = tf.keras.Input(shape = img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    #x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, 3, kernel_initializer = initializer, strides = 2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x #set aside residual

    # Blocks 1, 2, 3 are identical appart from the feature depth
    for filters in [64, 128, 256]:
        x = layers.Conv2D(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x) #probar max pooling(2, tal)
        x = layers.Dropout(_dropout)(x) #dropout after the pooling layers

        #project residual (?)
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        
        x = layers.add([x, residual]) #add back residual
        previous_block_activation = x #set aside next residual
    ###-----------------------------------
    #### [Middle]
    #x = layers.Conv2D(256, (3, 3), activation = "relu", padding="same")(x)
    #x = layers.BatchNormalization()(x)
   
    #x = layers.Conv2D(256, (3, 3), activation = "relu", padding="same")(x)
    #x = layers.BatchNormalization()(x)

    ###-----------------------------------
    ### [Second half of the network: upsampling inputs] ###
    for filters in [256, 128, 64, 32]:

        x = layers.Conv2DTranspose(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2DTranspose(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Dropout(_dropout)(x)
        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)   #change x and y shape
        residual = layers.Conv2D(filters, 1, padding="same")(residual) #change z shape
        x = layers.add([x, residual]) #Add back residual
        previous_block_activation = x #Set aside next residual

    # Add per-pixel classification layer

    #outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    outputs = layers.Conv2D(num_classes, 3, activation = "sigmoid", padding="same")(x)

    #Define model
    model = custom_model(inputs, outputs)


    #optimizer = keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer, 
                  loss = 'binary_crossentropy',
                  metrics = [
                             tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
                            ]
        )
        #optimizer = keras.optimizers.Adam()
            #model.compile(
            #optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
            #loss="sparse_categorical_crossentropy")
        #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        #loss = tf.nn.sigmoid_cross_entropy_with_logits
        #model.compile(
        #        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
        #        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        #    )
        ###momentum : Exponential average of the gradients

    return model

def get_PReLU_model(img_size, num_classes):
    """
    img_size follows numpy's format (y,x)
    """

    initializer = 'he_normal'
    _dropout = 0.2
    inputs = tf.keras.Input(shape = img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    #x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, 3, kernel_initializer = initializer, strides = 2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1,2])(x)
    # x = layers.ReLU()(x)

    previous_block_activation = x #set aside residual

    # Blocks 1, 2, 3 are identical appart from the feature depth
    for filters in [64, 128, 256]:
        x = layers.SeparableConv2D(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1,2])(x)
        # x = layers.ReLU()(x)

        x = layers.SeparableConv2D(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1,2])(x)
        # x = layers.ReLU()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x) #probar max pooling(2, tal)
        x = layers.Dropout(_dropout)(x) #dropout after the pooling layers

        #project residual (?)
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        
        x = layers.add([x, residual]) #add back residual
        previous_block_activation = x #set aside next residual
    ###-----------------------------------
    #### [Middle]
    #x = layers.Conv2D(256, (3, 3), activation = "relu", padding="same")(x)
    #x = layers.BatchNormalization()(x)
   
    #x = layers.Conv2D(256, (3, 3), activation = "relu", padding="same")(x)
    #x = layers.BatchNormalization()(x)

    ###-----------------------------------
    ### [Second half of the network: upsampling inputs] ###
    for filters in [256, 128, 64, 32]:

        x = layers.Conv2DTranspose(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1,2])(x)
        # x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1,2])(x)
        # x = layers.ReLU()(x)

        x = layers.Dropout(_dropout)(x)
        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)   #change x and y shape
        residual = layers.Conv2D(filters, 1, padding="same")(residual) #change z shape
        x = layers.add([x, residual]) #Add back residual
        previous_block_activation = x #Set aside next residual

    # Add per-pixel classification layer

    #outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    outputs = layers.Conv2D(num_classes, 3, activation = "sigmoid", padding="same")(x)

    #Define model
    model = custom_model(inputs, outputs)


    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                        initial_learning_rate = 0.002,
                        decay_steps = 35 * 639,
                        alpha = 0.2
                       )

#    tf.keras.optimizers.Adam(
#    learning_rate=0.001,
#    beta_1=0.9,
#    beta_2=0.999,
#    epsilon=1e-07,
#    amsgrad=False,
#    name="Adam",
#    **kwargs
#)


    #optimizer = keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer, 
                  loss = 'binary_crossentropy',
                  metrics = [
                             tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
                            ]
        )
        #optimizer = keras.optimizers.Adam()
            #model.compile(
            #optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
            #loss="sparse_categorical_crossentropy")
        #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        #loss = tf.nn.sigmoid_cross_entropy_with_logits
        #model.compile(
        #        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
        #        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        #    )
        ###momentum : Exponential average of the gradients

    return model


def get_PReLU_model2(img_size, num_classes):
    """
    img_size follows numpy's format (y,x)
    """

    initializer =  'he_normal' # 'random_normal' 
    _dropout = 0.2
    inputs = tf.keras.Input(shape = img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    #x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, 3, kernel_initializer = initializer, strides = 2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1,2])(x)
    # x = layers.ReLU()(x)

    previous_block_activation = x #set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth
    for filters in [64, 128, 256]:
        x = layers.SeparableConv2D(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1,2])(x)

        x = layers.SeparableConv2D(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)

        residual = layers.Conv2D(filters, 1, padding="same")(previous_block_activation)
        x = layers.add([x, residual]) #add back residual

        x = layers.PReLU(shared_axes=[1,2])(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x) #probar max pooling(2, tal)
        x = layers.Dropout(_dropout)(x) #dropout after the pooling layers

        #project residual (?)
        previous_block_activation = x #set aside next residual
    ###-----------------------------------
    #### [Middle]
    #x = layers.Conv2D(256, (3, 3), activation = "relu", padding="same")(x)
    #x = layers.BatchNormalization()(x)
   
    #x = layers.Conv2D(256, (3, 3), activation = "relu", padding="same")(x)
    #x = layers.BatchNormalization()(x)

    ###-----------------------------------
    ### [Second half of the network: upsampling inputs] ###
    for filters in [256, 128, 64, 32]:

        x = layers.Conv2DTranspose(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1,2])(x)

        x = layers.Conv2DTranspose(filters, 3, kernel_initializer = initializer, padding="same")(x)
        x = layers.BatchNormalization()(x)

        residual = layers.Conv2D(filters, 1, padding="same")(previous_block_activation)
        x = layers.add([x, residual]) #Add back residual

        x = layers.PReLU(shared_axes=[1,2])(x)

        x = layers.Dropout(_dropout)(x)
        x = layers.UpSampling2D(2)(x)

        # Project residual
        previous_block_activation = x #Set aside next residual

    # Add per-pixel classification layer

    #outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    outputs = layers.Conv2D(num_classes, 3, activation = "sigmoid", padding="same")(x)

    #Define model
    model = custom_model(inputs, outputs)


    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                        initial_learning_rate = initial_learning_rate,
                        decay_steps = epochs * steps_per_epoch,
                        alpha = 0.2
                       )

#    tf.keras.optimizers.Adam(
#    learning_rate=0.001,
#    beta_1=0.9,
#    beta_2=0.999,
#    epsilon=1e-07,
#    amsgrad=False,
#    name="Adam",
#    **kwargs
#)


    #optimizer = keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer, 
                  loss = 'binary_crossentropy',
                  metrics = [
                             tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
                            ]
        )
        #optimizer = keras.optimizers.Adam()
            #model.compile(
            #optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
            #loss="sparse_categorical_crossentropy")
        #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        #loss = tf.nn.sigmoid_cross_entropy_with_logits
        #model.compile(
        #        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
        #        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        #    )
        ###momentum : Exponential average of the gradients

    return model


def get_PReLU_model_old(img_size, num_classes):
    """
    img_size follows numpy's format (y,x)
    """
    inputs = tf.keras.Input(shape = img_size + (3,))
    _dropout = 0.7
    ### [First half of the network: downsampling inputs] ###

    #x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1,2])(x)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1,2])(x)

    previous_block_activation = x #set aside residual

    # Blocks 1, 2, 3 are identical appart from the feature depth
    for filters in [64, 128, 256]:
        x = layers.PReLU(shared_axes=[1,2])(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.PReLU(shared_axes=[1,2])(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.Dropout(_dropout)(x) #dropout after the pooling layers

        #project residual (?)
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation)
        
        x = layers.add([x, residual]) #add back residual
        previous_block_activation = x #set aside next residual
    ###-----------------------------------
    #### [Middle]
    #x = layers.Conv2D(256, (3, 3), activation = "relu", padding="same")(x)
    #x = layers.BatchNormalization()(x)
   
    #x = layers.Conv2D(256, (3, 3), activation = "relu", padding="same")(x)
    #x = layers.BatchNormalization()(x)

    ###-----------------------------------
    ### [Second half of the network: upsampling inputs] ###
    for filters in [256, 128, 64, 32]:
        x = layers.PReLU(shared_axes=[1,2])(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.PReLU(shared_axes=[1,2])(x)
        x = layers.Convolution2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dropout(_dropout)(x)
        x = layers.UpSampling2D(2)(x)
       
        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)   #change x and y shape
        residual = layers.Conv2D(filters, 1, padding="same")(residual) #change z shape
        x = layers.add([x, residual]) #Add back residual
        previous_block_activation = x #Set aside next residual

    # Add per-pixel classification layer

    #outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    outputs = layers.Conv2D(num_classes, 3, activation = "sigmoid", padding="same")(x)

    #Define model
    model = custom_model(inputs, outputs)


    #optimizer = keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer, 
                  loss = 'binary_crossentropy',
                  metrics = [
                             tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
                            ]
        )
        #optimizer = keras.optimizers.Adam()
            #model.compile(
            #optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
            #loss="sparse_categorical_crossentropy")
        #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        #loss = tf.nn.sigmoid_cross_entropy_with_logits
        #model.compile(
        #        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
        #        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        #    )
        ###momentum : Exponential average of the gradients

    return model

#based on: https://www.kaggle.com/phoenigs/u-net-dropout-augmentation-stratification

###TODO:: ADD BATCH NORMLIZATION
def get_model_with_skip_connections_UNET1(img_size, num_classes): 
    """
    img_size follows numpy's format (y,x)
    """
    inputs = tf.keras.Input(shape = img_size + (3,))
    _dropout = 0.1

    ### [First half of the network: downsampling inputs] ###
    # 3 -> 32 (depth channels)
    conv1 = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)

    conv1 = layers.Conv2D(32, (3, 3), padding="same")(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)

    pool1 = layers.MaxPooling2D((2,2))(conv1)

    pool1 = layers.Dropout(_dropout)(pool1)
    #-------------------------------------------------------------

    # 32 -> 64 (depth channels)
    conv2 = layers.Conv2D(64, (3, 3), padding="same")(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)

    conv2 = layers.Conv2D(64, (3, 3), padding="same")(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)

    pool2 = layers.MaxPooling2D((2,2))(conv2)

    pool2 = layers.Dropout(_dropout)(pool2)
    #-------------------------------------------------------------

    # 64 -> 128 (depth channels)
    conv3 = layers.Conv2D(128, (3, 3), padding="same")(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)

    conv3 = layers.Conv2D(128, (3, 3), padding="same")(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)

    pool3 = layers.MaxPooling2D((2,2))(conv3)

    pool3 = layers.Dropout(_dropout)(pool3)

    #-------------------------------------------------------------

    convm = layers.Conv2D(256, (3, 3), padding="same")(pool3)
    convm = layers.BatchNormalization()(convm)
    convm = layers.Activation("relu")(convm)

    convm = layers.Conv2D(256, (3, 3), padding="same")(convm)
    convm = layers.BatchNormalization()(convm)
    convm = layers.Activation("relu")(convm)

    convm = layers.Dropout(_dropout)(convm)

    #-------------------------------------------------------------

    # 256 -> 128 (depth channels)
    deconv3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = layers.concatenate([deconv3, conv3])
       
    uconv3 = layers.Conv2D(128, (3, 3), padding="same")(uconv3)
    uconv3 = layers.BatchNormalization()(uconv3)
    uconv3 = layers.Activation("relu")(uconv3)

    uconv3 = layers.Conv2D(128, (3, 3), padding="same")(uconv3)
    uconv3 = layers.BatchNormalization()(uconv3)
    uconv3 = layers.Activation("relu")(uconv3)

    uconv3 = layers.Dropout(_dropout)(uconv3)
    #-------------------------------------------------------------

    # 128 -> 64 (depth channels) 
    deconv2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
       
    uconv2 = layers.Conv2D(64, (3, 3), padding="same")(uconv2)
    uconv2 = layers.BatchNormalization()(uconv2)
    uconv2 = layers.Activation("relu")(uconv2)

    uconv2 = layers.Conv2D(64, (3, 3), padding="same")(uconv2)
    uconv2 = layers.BatchNormalization()(uconv2)
    uconv2 = layers.Activation("relu")(uconv2)

    uconv2 = layers.Dropout(_dropout)(uconv2)
    #-------------------------------------------------------------

    # 64 -> 32 (depth channels)
    deconv1 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
       
    uconv1 = layers.Conv2D(32, (3, 3), padding="same")(uconv1)
    uconv1 = layers.BatchNormalization()(uconv1)
    uconv1 = layers.Activation("relu")(uconv1)

    uconv1 = layers.Dropout(_dropout)(uconv1)

    uconv1 = layers.Conv2D(32, (3, 3), padding="same")(uconv1)
    uconv1 = layers.BatchNormalization()(uconv1)
    uconv1 = layers.Activation("relu")(uconv1)

    #-------------------------------------------------------------
    #-------------------------------------------------------------
    # Add per-pixel classification layer
    outputs = layers.Conv2D(num_classes, (1,1), activation = "sigmoid", padding="same")(uconv1)

    #Define model
    model = custom_model(inputs, outputs)
    model.compile(optimizer = tf.keras.optimizers.Adam(),
                  loss = 'binary_crossentropy',
                  metrics = [
                             tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
                            ]
    )

    return model
    
    #model.compile(keras.optimizers.SGD(lr=0.01, decay = 1e-6, momentum=0.9, clipnorm=1.0, nesterov = True),
    #              loss = 'binary_crossentropy',
    #              metrics = ['accuracy'] #, tf.keras.metrics.MeanIoU(num_classes=2)] #, ]
    #    )
        #optimizer = keras.optimizers.Adam()
            #model.compile(
            #optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
            #loss="sparse_categorical_crossentropy")
        #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        #loss = tf.nn.sigmoid_cross_entropy_with_logits
        #model.compile(
        #        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
        #        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        #    )
        ###momentum : Exponential average of the gradients

def get_model_with_skip_connections_UNET0(img_size, num_classes): 
    """
    img_size follows numpy's format (y,x)
    """
    inputs = tf.keras.Input(shape = img_size + (3,))
    _dropout = 0.4
    #initializer = tf.keras.initializers.RandomNormal(stddev=0.06)

    ### [First half of the network: downsampling inputs] ###
    # 3 -> 32 (depth channels)
    initializer = tf.keras.initializers.RandomNormal(stddev=0.1)
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(inputs)
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(conv1)

    pool1 = layers.MaxPooling2D((2,2))(conv1)
    pool1 = layers.Dropout(_dropout)(pool1)
    #-------------------------------------------------------------

    # 32 -> 64 (depth channels)
    initializer = tf.keras.initializers.RandomNormal(stddev=0.083)
    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(pool1)
    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(conv2)

    pool2 = layers.MaxPooling2D((2,2))(conv2)
    pool2 = layers.Dropout(_dropout)(pool2)
    #-------------------------------------------------------------

    # 64 -> 128 (depth channels)
    initializer = tf.keras.initializers.RandomNormal(stddev=0.059)
    conv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(pool2)
    conv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(conv3)

    pool3 = layers.MaxPooling2D((2,2))(conv3)
    pool3 = layers.Dropout(_dropout)(pool3)

    #-------------------------------------------------------------

    initializer = tf.keras.initializers.RandomNormal(stddev=0.0416)
    convm = layers.Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(pool3)
    convm = layers.Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(convm)

    #-------------------------------------------------------------

    # 256 -> 128 (depth channels)
    initializer = tf.keras.initializers.RandomNormal(stddev=0.0295)
    deconv3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer = initializer)(convm)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(_dropout)(uconv3)

    uconv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(uconv3)
    uconv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(uconv3)



    #-------------------------------------------------------------

    # 128 -> 64 (depth channels) 
    initializer = tf.keras.initializers.RandomNormal(stddev=0.0416)
    deconv2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer = initializer)(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(_dropout)(uconv2)
    
    uconv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(uconv2)
    uconv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(uconv2)

    #-------------------------------------------------------------

    # 64 -> 32 (depth channels)
    initializer = tf.keras.initializers.RandomNormal(stddev=0.059)
    deconv1 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer = initializer)(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(_dropout)(uconv1)

    uconv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(uconv1)
    uconv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_initializer = initializer)(uconv1)


    #-------------------------------------------------------------
    #-------------------------------------------------------------
    # Add per-pixel classification layer
    initializer = tf.keras.initializers.RandomNormal(stddev=0.083)
    outputs = layers.Conv2D(num_classes, (1,1), activation = "sigmoid", padding="same", kernel_initializer = initializer)(uconv1)

    #Define model
    model = custom_model(inputs, outputs)
    model.compile(optimizer = tf.keras.optimizers.Adam(),
                  loss = 'binary_crossentropy',
                  metrics = [
                             tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
                            ]
    )

    return model
    
    #model.compile(keras.optimizers.SGD(lr=0.01, decay = 1e-6, momentum=0.9, clipnorm=1.0, nesterov = True),
    #              loss = 'binary_crossentropy',
    #              metrics = ['accuracy'] #, tf.keras.metrics.MeanIoU(num_classes=2)] #, ]
    #    )
        #optimizer = keras.optimizers.Adam()
            #model.compile(
            #optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
            #loss="sparse_categorical_crossentropy")
        #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        #loss = tf.nn.sigmoid_cross_entropy_with_logits
        #model.compile(
        #        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
        #        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        #    )
        ###momentum : Exponential average of the gradients

###TODO:: ADD BATCH NORMLIZATION
def get_model_with_skip_connections_and_residuals(img_size, num_classes): 
    """
    img_size follows numpy's format (y,x)
    """
    inputs = tf.keras.Input(shape = img_size + (3,))
    _dropout = 0.25

    ### [First half of the network: downsampling inputs] ###
    # 3 -> 32 (depth channels)
    conv1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D((2,2))(conv1)
    pool1 = layers.Dropout(_dropout)(pool1)

    #-------------------------------------------------------------
    # 32 -> 64 (depth channels)
    conv2 = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D((2,2))(conv2)
    pool2 = layers.Dropout(_dropout)(pool2)
    # Project residual
    residual2 = layers.SeparableConv2D(64, (1,1), strides=2)(pool1)   #change x and y shape
    block_out2 = layers.add([pool2, residual2]) #Add back residual

    #-------------------------------------------------------------
    # 64 -> 128 (depth channels)
    conv3 = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(block_out2)
    conv3 = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling2D((2,2))(conv3)
    pool3 = layers.Dropout(_dropout)(pool3)
    # Project residual
    residual3 = layers.SeparableConv2D(128, (1,1), strides=2)(block_out2)   #change x and y shape
    block_out3 = layers.add([pool3, residual3]) #Add back residual
    #-------------------------------------------------------------

    # 128 -> 256 (depth channels)
    conv4 = layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same")(block_out3)
    conv4 = layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = layers.MaxPooling2D((2,2))(conv4)
    pool4 = layers.Dropout(_dropout)(pool4)
    # Project residual
    residual4 = layers.SeparableConv2D(256, (1,1), strides=2)(block_out3)   #change x and y shape
    block_out4 = layers.add([pool4, residual4]) #Add back residual
    #-----------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------
    
    # Middle
    convm = layers.SeparableConv2D(512, (3, 3), activation="relu", padding="same")(block_out4)
    convm = layers.SeparableConv2D(512, (3, 3), activation="relu", padding="same")(convm)
    #-----------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------

    # 512 -> 256 (depth channels)
    deconv4 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(_dropout)(uconv4)
    uconv4 = layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = layers.SeparableConv2D(256, (3, 3), activation="relu", padding="same")(uconv4)
    # Project residual
    uresidual4 = layers.UpSampling2D(2)(convm)   #change x and y shape
    uresidual4 = layers.SeparableConv2D(256, 1, padding="same")(uresidual4) #change z shape
    ublock_out4 = layers.add([uconv4, uresidual4]) #Add back residual
    #-------------------------------------------------------------

    # 256 -> 128 (depth channels)
    deconv3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(ublock_out4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(_dropout)(uconv3)
    uconv3 = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(uconv3)
    # Project residual
    uresidual3 = layers.UpSampling2D(2)(ublock_out4)   #change x and y shape
    uresidual3 = layers.SeparableConv2D(128, 1, padding="same")(uresidual3) #change z shape
    ublock_out3 = layers.add([uconv3, uresidual3]) #Add back residual
    #-------------------------------------------------------------

    # 128 -> 64 (depth channels) 
    deconv2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(ublock_out3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(_dropout)(uconv2)
    uconv2 = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(uconv2)
    # Project residual
    uresidual2 = layers.UpSampling2D(2)(ublock_out3)   #change x and y shape
    uresidual2 = layers.SeparableConv2D(64, 1, padding="same")(uresidual2) #change z shape
    ublock_out2 = layers.add([uconv2, uresidual2]) #Add back residual
    #-------------------------------------------------------------

    # 64 -> 32 (depth channels)
    deconv1 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(ublock_out2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(_dropout)(uconv1)
    uconv1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(uconv1)
    # Project residual
    uresidual1 = layers.UpSampling2D(2)(ublock_out2)   #change x and y shape
    uresidual1 = layers.Conv2D(32, 1, padding="same")(uresidual1) #change z shape
    ublock_out1 = layers.add([uconv1, uresidual1]) #Add back residual
    #-------------------------------------------------------------
    #-------------------------------------------------------------


    # Add per-pixel classification layer
    outputs = layers.Conv2D(num_classes, (1,1), activation = "sigmoid", padding="same")(ublock_out1)

    #Define model
    model = custom_model(inputs, outputs)

    model.compile(optimizer = tf.keras.optimizers.Adam(),
                  loss = 'binary_crossentropy',
                  metrics = [
                             tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
                            ]
     )
        #optimizer = keras.optimizers.Adam()
            #model.compile(
            #optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
            #loss="sparse_categorical_crossentropy")
        #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        #loss = tf.nn.sigmoid_cross_entropy_with_logits
        #model.compile(
        #        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
        #        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
        #    )
        ###momentum : Exponential average of the gradients

    return model


if __name__ == "__main__":
    # Free up RAM in case the model definition cells were run multiple times
    tf.keras.backend.clear_session()

    # Build model
    model = get_PReLU_model2(img_size, NUM_CLASSES)
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, to_file="./gen/" + model_name + ".png")

    color_map = defaultdict(dict)
    color_map[layers.Conv2D]['fill'] = '#fb5607'
    color_map[layers.Conv2DTranspose]['fill'] = '#fb5607'
    color_map[layers.MaxPooling2D]['fill'] = '#8338ec'
    color_map[layers.UpSampling2D]['fill'] = '#8338ec'
    color_map[layers.Dropout]['fill'] = '#03045e'
    color_map[layers.Add]['fill'] = '#00f5d4'

    plt.imshow(visualkeras.layered_view(model, legend=True, type_ignore=[layers.BatchNormalization, layers.Activation], draw_volume=False, color_map=color_map))
    plt.show()

    print("Done")




#class CNN_Block(layers.layers):
#    def __init__(self, out_channels, kernel_size=3, dropout=0.0, padding='same'):
#        super(CNN_Block, self).__init__()
#        self.conv = layers.conv2D(out_channels, kernel_size, padding=padding)
#        self.bn = layers.BatchNormalization()
#        self.dropout = layers.Dropout(dropout)

#    def call(self, input_tensor, training=False):
#        x = self.conv(input_tensor)
#        x = self.bn(x, training=training)
#        x = layers.Activation('relu')(x)
#        x = self.dropout(x)
#        return x

#def get_simple_FCN_model(img_size, num_classes):
#    filters = [32, 64, 128, 256, 256, 128, 64, 32]
#    model = tf.keras.Sequential([
#            CNN_Block(filter) for filter in filters
#        ].append(layers.Conv2D(num_classes, 3, activation = "sigmoid", padding="same"))
#    )
#    model.compile(
#        tf.keras.optimizers.Adam(),   
#        loss = 'binary_crossentropy',
#        metrics = [tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)]
#    )