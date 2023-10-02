import numpy as np
from PIL import Image
import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.keras import models 
import os
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO
)


# setup dir
if os.path.exists('output'):
  pass
else:
  os.mkdir('output')

def load_img(path_to_img, model):
  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = tf.keras.utils.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  
  if model == 'vgg19':
    # Preprocess the input image for VGG19 model
    img = tf.keras.applications.vgg19.preprocess_input(img)
  elif model == 'vgg16':
    # Preprocess the input image for VGG16 model
    img = tf.keras.applications.vgg16.preprocess_input(img)
  
  return img


# prepare the data

def load_and_process_img(path_to_img, model):
  img = load_img(path_to_img, model)
  
  return img

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers_vgg19 = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

style_layers_vgg16 = ['block1_conv1',
    'block1_conv2',
    'block2_conv1',
    'block2_conv2',
    'block3_conv1',
    'block3_conv2']

num_content_layers = len(content_layers)
num_style_layers_vgg19 = len(style_layers_vgg19)
num_style_layers_vgg16 = len(style_layers_vgg16)



# building the model

def get_model(model):
  # Load our model. We load pretrained VGG, trained on imagenet data
  if model == 'vgg19':
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  # vgg 16
  elif model == 'vgg16':
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet') 
  
  vgg.trainable = False
  # Get output layers corresponding to style and content layers 
  if model == 'vgg19':
    style_outputs = [vgg.get_layer(name).output for name in style_layers_vgg19]
  elif model == 'vgg16':
    style_outputs = [vgg.get_layer(name).output for name in style_layers_vgg16]
    
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  # Build model 
  return models.Model(vgg.input, model_outputs)

# content loss 
def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))

# def compute loss

def gram_matrix(input_tensor):
  # We make the image channels first 
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
  # height, width, num filters of each layer
  # We scale the loss at a given layer by the size of the feature map and the number of filters
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)


# run gradient descent

def get_feature_representations(model, content_path, style_path):
  # Load our images in 
  content_image = load_and_process_img(content_path, model)
  style_image = load_and_process_img(style_path, model)
  
  # batch compute content and style features
  style_outputs = model(style_image)
  content_outputs = model(content_image)
  
  
  # Get the style and content feature representations from our model  
  if model == 'vgg19':
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers_vgg19]]
  elif model == 'vgg16':
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers_vgg16]]
    
  content_features = [content_layer[0] for content_layer in content_outputs[num_content_layers:]]
  return style_features, content_features

  
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
  style_weight, content_weight = loss_weights
  
  # Feed our init image through our model. This will give us the content and 
  # style representations at our desired layers. Since we're using eager
  # our model is callable just like any other function!
  model_outputs = model(init_image)
  
  if model == 'vgg19':
    style_output_features = model_outputs[:num_style_layers_vgg19]
  elif model == 'vgg16':
    style_output_features = model_outputs[:num_style_layers_vgg16]
    
  content_output_features = model_outputs[num_content_layers:]
  
  style_score = 0
  content_score = 0

  # Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
  weight_per_style_layer = 1.0 / float(num_style_layers_vgg19 if model == 'vgg19' else num_style_layers_vgg16)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
  # Accumulate content losses from all layers 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
  style_score *= style_weight
  content_score *= content_weight

  # Get total loss
  loss = style_score + content_score 
  return loss, style_score, content_score

def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss



def run_style_transfer(content_path, 
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3, 
                       style_weight=1e-2,
                       model='vgg19'): 
  logging.info(f'starting style transfer for {content_path}')
  # log the params
  logging.info(f'content weight: {content_weight}')
  logging.info(f'style weight: {style_weight}')
  logging.info(f'style layers: {style_layers_vgg19 if model == "vgg19" else style_layers_vgg16}')
  logging.info(f'num iterations: {num_iterations}')
  logging.info(f'model: {model}')
  
  # We don't need to (or want to) train any layers of our model, so we set their
  # trainable to false. 
  model = get_model(model) 
  for layer in model.layers:
    layer.trainable = False
  
  # Get the style and content feature representations (from our specified intermediate layers) 
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  # Set initial image
  init_image = load_and_process_img(content_path, model)
  init_image = tf.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  opt = tf.optimizers.Adam(learning_rate=5, epsilon=1e-1)

  
  # Store our best result
  best_loss, best_img = float('inf'), None
  
  # Create a nice config 
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }
    
  # For displaying
  num_rows = 2
  num_cols = 5

  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  for i in tqdm(range(num_iterations)):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    
    if loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())
  logging.info('Total time: {:.4f}s'.format(time.time() - global_start))

      
  return best_img, best_loss
