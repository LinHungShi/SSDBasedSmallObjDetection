#Embedded file name: /home/geniussport/Detection/context_ssd.py
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Add, Activation, Conv2D, MaxPooling2D, Reshape, Concatenate, BatchNormalization, Conv2DTranspose
from anchor_box import AnchorBoxes
from keras.initializers import Constant
import tensorflow as tf
constant = 1
f1 = 48
f2 = 64

def imgNorm(image):
    mean, var = tf.nn.moments(image, axes=[1, 2], keep_dims=True)
    return (image - mean) / tf.sqrt(var)


def context_ssd(image_size, n_classes, scales = None, aspect_ratios = [1.0 / 3.0, 0.5,1.0,2.0,3.0], 
    two_boxes_for_ar1 = True, limit_boxes = False, variances = [1,1,1,1], coords = 'centroids', normalize_coords = False):
    
    print 'Creating network...'
    n_predictor_layers = 1
    if aspect_ratios is None:
        raise ValueError('aspect_ratios cannot both be None')
    if scales is None:
        raise ValueError('scales need to be specified')
    if len(scales) != 2:
        raise ValueError('scale must be 2')
    if len(variances) != 4:
        raise ValueError('4 variance values musr be pased')
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError('All variances must be larger than 0')
    if (1 in aspect_ratios) & two_boxes_for_ar1:
        n_boxes = len(aspect_ratios) + 1
    else:
        n_boxes = len(aspect_ratios)
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    x = Input((img_height, img_width, img_channels))
    normed = Lambda(imgNorm, output_shape=(img_height, img_width, img_channels), name='lambda1')(x)
    conv1_1 = Conv2D(f1, (3, 3), activation='relu', padding='same', name='conv1_1', bias_initializer=Constant(constant))(normed)
    conv1_2 = Conv2D(f1, (3, 3), activation='relu', padding='same', name='conv1_2', bias_initializer=Constant(constant))(conv1_1)
    add1 = Add(name='add1')([conv1_1, conv1_2])
    conv1_3 = Conv2D(f1, (3, 3), activation='relu', strides=(2, 2), padding='same', name='conv1_3', bias_initializer=Constant(constant))(add1)
    conv2_1 = Conv2D(f1, (3, 3), activation='relu', padding='same', name='conv2_1', bias_initializer=Constant(constant))(conv1_3)
    conv2_2 = Conv2D(f1, (3, 3), activation='relu', padding='same', name='conv2_2', bias_initializer=Constant(constant))(conv2_1)
    add2 = Add(name='add2')([conv1_3, conv2_2])
    conv2_3 = Conv2D(f1, (3, 3), activation='relu', strides=(2, 2), padding='same', name='conv2_3', bias_initializer=Constant(constant))(add2)
    conv3_1 = Conv2D(f1, (3, 3), activation='relu', padding='same', name='conv3_1', bias_initializer=Constant(constant))(conv2_3)
    conv3_2 = Conv2D(f1, (3, 3), activation='relu', padding='same', name='conv3_2', bias_initializer=Constant(constant))(conv3_1)
    add3 = Add(name='add3')([conv2_3, conv3_2])
    conv3_3 = Conv2D(f2, (3, 3), activation='relu', strides=(2, 2), padding='same', name='conv3_3', bias_initializer=Constant(constant))(add3)
    conv4_1 = Conv2D(f2, (3, 3), activation='relu', padding='same', name='conv4_1', bias_initializer=Constant(constant))(conv3_3)
    conv4_2 = Conv2D(f2, (3, 3), activation='relu', padding='same', name='conv4_2', bias_initializer=Constant(constant))(conv4_1)
    add4 = Add(name='add4')([conv3_3, conv4_2])
    conv4_3 = Conv2D(f2, (3, 3), activation='relu', strides=(2, 2), padding='same', name='conv4_3', bias_initializer=Constant(constant))(add4)
    conv4_3 = BatchNormalization(name='bn4', momentum=0.5)(conv4_3)
    context4_2s = Conv2D(f2, (3, 3), activation='relu', strides=(1, 1), dilation_rate=(2, 2), padding='same', name='context4_2s', bias_initializer=Constant(constant))(conv4_3)
    context4_3s = Conv2D(f2, (3, 3), activation='relu', strides=(1, 1), dilation_rate=(3, 3), padding='same', name='context4_3s', bias_initializer=Constant(constant))(conv4_3)
    context4_4s = Conv2D(f2, (3, 3), activation='relu', strides=(1, 1), dilation_rate=(4, 4), padding='same', name='context4_4s', bias_initializer=Constant(constant))(conv4_3)
    context4_concatenate = Concatenate(axis=-1, name='context4_concatenate')([context4_2s, context4_3s, context4_4s])
    context4_agg = Conv2D(f2, (1, 1), activation='relu', strides=(1, 1), padding='same', name='context4_agg', bias_initializer=Constant(constant))(context4_concatenate)
    upsampling4 = Conv2DTranspose(f2, (1, 1), activation='relu', strides=(2, 2), padding='same', name='upsampling4', bias_initializer=Constant(constant))(context4_agg)
    add_up4_conv3_3 = Concatenate(axis=-1, name='add_up4_conv3_3')([upsampling4, conv3_3])
    add_up4_conv3_3 = BatchNormalization()(add_up4_conv3_3)
    context4_l2_norm = Conv2DTranspose(f2, (1, 1), activation='relu', strides=(2, 2), padding='same', name='test', bias_initializer=Constant(constant))(add_up4_conv3_3)
    context4_l2_norm = Concatenate(axis=-1)([conv2_3, context4_l2_norm])
    context4_l2_norm = BatchNormalization()(context4_l2_norm)
    mbox_fc_conv = Conv2D(f2, (3, 3), activation='relu', strides=(1, 1), padding='same', name='mbox_fc_conf', bias_initializer=Constant(constant))(context4_l2_norm)
    mbox_pre_conv = Conv2D(f2, (3, 3), activation='relu', strides=(1, 1), padding='same', name='mbox_pre_conf', bias_initializer=Constant(constant))(mbox_fc_conv)
    mbox_conf = Conv2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding='same', name='mbox_conf', bias_initializer=Constant(constant))(mbox_pre_conv)
    mbox_fc_loc = Conv2D(f2, (3, 3), activation='relu', strides=(1, 1), padding='same', name='mbox_fc_loc', bias_initializer=Constant(constant))(context4_l2_norm)
    mbox_pre_loc = Conv2D(f2, (3, 3), activation='relu', padding='same', name='mbox_pre_loc', bias_initializer=Constant(constant))(mbox_fc_loc)
    mbox_loc = Conv2D(n_boxes * 4, (3, 3), padding='same', name='mbox_loc', bias_initializer=Constant(constant))(mbox_pre_loc)
    mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios, two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='mbox_priorbox')(mbox_loc)
    mbox_conf_reshape = Reshape((-1, n_classes), name='mbox_conf_reshape')(mbox_conf)
    mbox_loc_reshape = Reshape((-1, 4), name='mbox_loc_reshape')(mbox_loc)
    mbox_priorbox_reshape = Reshape((-1, 8), name='mbox_priorbox_reshape')(mbox_priorbox)
    mbox_conf_softmax = Activation('softmax', name='conf_softmax')(mbox_conf_reshape)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc_reshape, mbox_priorbox_reshape])
    model = Model(inputs=x, outputs=predictions)
    predictor_sizes = np.array([mbox_conf._keras_shape[1:3]])
    return (model, predictor_sizes)
