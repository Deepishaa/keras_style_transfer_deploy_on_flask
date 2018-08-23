from keras.models import Model
from keras.layers import MaxPooling2D, Conv2D, Input
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from keras.applications import vgg19

def customVGG16(input_tensor):
    vgg16 = vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
    input_shape = (224, 224, 3)
    if not K.is_keras_tensor(input_tensor):
        img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor

    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # x = Flatten(name='flatten')(x)
    # x = Dense(512, activation='relu', name='fc1')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(256, activation='relu', name='fc2')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(classes, activation='softmax', name='final_output')(x)
    
    # model = Model(img_input, x, name='flag')
    # return model


    # model = Sequential()
    x = Conv2D(64, (3, 3), dtype = 'float32', input_shape=input_shape, padding='same',
           activation='relu', weights=vgg16.layers[1].get_weights())(img_input)#--> layer 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', weights=vgg16.layers[2].get_weights())(x)#--> layer 2
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), weights=vgg16.layers[3].get_weights())(x)#--> layer 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', weights=vgg16.layers[4].get_weights())(x)#--> layer 4
    x = Conv2D(128, (3, 3), activation='relu', padding='same', weights=vgg16.layers[5].get_weights())(x)#--> layer 5
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), weights=vgg16.layers[6].get_weights())(x)#--> layer 6
    x = Conv2D(256, (3, 3), activation='relu', padding='same', weights=vgg16.layers[7].get_weights())(x)#--> layer 7
    x = Conv2D(256, (3, 3), activation='relu', padding='same', weights=vgg16.layers[8].get_weights())(x) #--> layer 8. 
    x = Conv2D(256, (3, 3), activation='relu', padding='same', weights=vgg16.layers[9].get_weights())(x)#--> layer 9
    x = Conv2D(256, (3, 3), activation='relu', padding='same', weights=vgg16.layers[10].get_weights())(x)#--> layer 9
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), weights=vgg16.layers[11].get_weights())(x)#--> layer 10
    x = Conv2D(512, (3, 3), activation='relu', padding='same', weights=vgg16.layers[12].get_weights())(x)#--> layer 11
    x = Conv2D(512, (3, 3), activation='relu', padding='same', weights=vgg16.layers[13].get_weights())(x)#--> layer 12
    x = Conv2D(512, (3, 3), activation='relu', padding='same', weights=vgg16.layers[14].get_weights())(x)#--> layer 13
    x = Conv2D(512, (3, 3), activation='relu', padding='same', weights=vgg16.layers[15].get_weights())(x)#--> layer 13
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), weights=vgg16.layers[16].get_weights())(x)#--> layer 14
    x = Conv2D(512, (3, 3), activation='relu', padding='same',  weights=vgg16.layers[17].get_weights())(x)#--> layer 15
    x = Conv2D(512, (3, 3), activation='relu', padding='same',  weights=vgg16.layers[18].get_weights())(x)#--> layer 16
    x = Conv2D(512, (3, 3), activation='relu', padding='same',  weights=vgg16.layers[19].get_weights())(x)#--> layer 17
    x = Conv2D(512, (3, 3), activation='relu', padding='same',  weights=vgg16.layers[20].get_weights())(x)#--> layer 17
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), weights=vgg16.layers[21].get_weights())(x)#--> layer 18

    model = Model(img_input, x, name='vgg16')
    return model

def load_image(x, target_size=None):
    img = pil_image.open(x)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        img = img.resize(width_height_tuple, pil_image.NEAREST)
    return img

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    
    data_format = tf.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=backend.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x    

def calc_rowsandcols(width, height):
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)
    return img_nrows,img_ncols

def preprocess_image(image_path, width, height):
    # dimensions of the generated picture.
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)
    img = load_image(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_input(x):
    data_format = K.image_data_format()
    x = x.astype(K.floatx(), copy=False)
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x

def deprocess_image(x,img_nrows, img_ncols):

    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    #add to respectice depths i.e 0,1,2
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# the gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is contentd on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination, width, height):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def content_loss(content, combination):
    return K.sum(K.square(combination - content))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x,width,height):
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def eval_loss_and_grads(x, width, height, f_outputs):
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

def save_image(x,fname):
    pil_image.fromarray(x).save(fname)


