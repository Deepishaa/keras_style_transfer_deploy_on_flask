import os
import argparse
import functions
import numpy as np
# from keras.layers import Dense, Activation, Dropout, Flatten
# from keras.engine import Input
from scipy.optimize import fmin_l_bfgs_b
import time
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('content_image_path', metavar='content', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')         
args = parser.parse_args()
content_image_path = args.content_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# dimensions of the generated picture.
width, height = functions.load_image(content_image_path).size

# get tensor representations of our images
content_image = K.variable(functions.preprocess_image(content_image_path, width, height))
style_reference_image = K.variable(functions.preprocess_image(style_reference_image_path,width, height))

# this will contain our generated image

img_nrows, img_ncols = functions.calc_rowsandcols(width,height)
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([content_image,
                              style_reference_image,
                              combination_image], axis=0)

# build the VGG16 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model = functions.customVGG16(input_tensor)
#print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions


# an auxiliary loss function
# designed to maintain the "content" of the
# content image in the generated image



# combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict['conv2d_12']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * functions.content_loss(content_image_features,
                                      combination_features)

feature_layers = ['conv2d_1', 'conv2d_3',
                  'conv2d_5', 'conv2d_8',
                  'conv2d_11']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = functions.style_loss(style_reference_features, combination_features,width,height)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * functions.total_variation_loss(combination_image,width,height)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = functions.eval_loss_and_grads(x, width, height, f_outputs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

def main():
    pass

if __name__ == "__main__":
    main()


x = functions.preprocess_image(content_image_path,width,height)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    
    print('Current loss value:', min_val)
    # save current generated image
    img = functions.deprocess_image(x.copy(),img_nrows, img_ncols)
    fname = result_prefix + '_at_iteration_%d.png' % i
    functions.save_image(img,fname)
    # image.save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))