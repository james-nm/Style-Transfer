# https://www.tensorflow.org/tutorials/generative/style_transfer
# https://arxiv.org/pdf/1508.06576.pdf

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time

import os

class style_processor:
  
  def __init__(self, content_path, style_path, **kwargs):

    # Clear any existing tf session
    self.reset()
    
    # Configure settings; unpack from kwargs
    # TODO: refactor input validation
    self.style_weight = kwargs.get('style_weight', 1e-2)
    assert(self.style_weight >= 0)
    
    self.content_weight = kwargs.get('content_weight', 1e4)
    assert(self.content_weight >= 0)
    
    self.total_variation_weight = kwargs.get('total_variation_weight', 30)
    assert(self.total_variation_weight >= 0)
    
    self.epochs = kwargs.get('epochs', 5)
    assert(self.epochs >= 0)
    
    self.steps_per_epoch = kwargs.get('steps_per_epoch', 100)
    assert(self.steps_per_epoch >= 0)

    # Load images from path
    self.content_image = self.load_img(content_path)
    self.style_image = self.load_img(style_path)

    x = tf.keras.applications.vgg19.preprocess_input(self.content_image*255)
    x = tf.image.resize(x, (224, 224)) # TODO: look into input aspect ratio
    
    # Content layer where will pull our feature maps
    self.content_layers = ['block5_conv2']
    
    # Style layer of interest
    self.style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    
    self.num_content_layers = len(self.content_layers)
    self.num_style_layers = len(self.style_layers)
    
    
  def process(self, verbose=True):
    '''
    Desired return for process (list):
      - stylized image
      - original content image (resized)
      - original style image (resized)
      - FUTURE: composite images + settings details
    '''
    ret = []
    
    self.extractor = StyleContentModel(self.style_layers, self.content_layers)
  
    self.results = self.extractor(tf.constant(self.content_image))
    
    self.style_targets = self.extractor(self.style_image)['style']
    self.content_targets = self.extractor(self.content_image)['content']
    
    self.optimizer = tf.optimizers.Adam(
        learning_rate=0.02, beta_1=0.99, epsilon=1e-1
    )
    image = tf.Variable(self.content_image)
  
    #run optimization
    start = time.time()
    
    step = 0
    for n in range(self.epochs):
      for m in range(self.steps_per_epoch):
        step += 1
        self.train_step(image)
        if verbose:
          print(".", end='')
      if verbose:
        display.clear_output(wait=True)
        display.display(self.tensor_to_image(image))
      
      print("Train step: {}".format(step))
    if verbose:
      end = time.time()
      print("Total time: {:.1f}".format(end-start))
    
    ret.append(self.tensor_to_image(image))
    ret.append(self.content_image)
    ret.append(self.style_image)
    
    return ret
    
  def reset(self):
    # Destroy the current TF graph for next session
    # TODO: can this be refactored into __init__?
    tf.keras.backend.clear_session()
  
  def tensor_to_image(self, tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0]  == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
  
  def load_img(self, path_to_img):
    # TODO: Add option for image resizing resolution
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img
  
  def imshow(self, image, title=None):
    if len(image.shape) > 3:
      image = tf.squeeze(image, axis=0)
      
    plt.imshow(image)
    if title:
      plt.title(title)
      
  def clip_0_1(self, image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
  
  def style_content_loss(self,outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2)
            for name in style_outputs.keys()
        ])
    style_loss *= self.style_weight / self.num_style_layers
    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2)
            for name in content_outputs.keys()
        ])
    
    content_loss *= self.content_weight / self.num_content_layers
    loss = style_loss + content_loss
    return loss
  
  @tf.function()
  def train_step(self,image):
    with tf.GradientTape() as tape:
      outputs = self.extractor(image)
      loss = self.style_content_loss(outputs)
      #TODO: add option for switching on/off regularization
      loss += self.total_variation_weight*tf.image.total_variation(image) #regularize loss with total_variation()
      
    grad = tape.gradient(loss, image)
    self.optimizer.apply_gradients([(grad, image)]) # optimizer
    image.assign(self.clip_0_1(image))

# Build a model that returns the style and content tensors
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    #tf.keras.models.Model.__init__(self)
    self.vgg = self.vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False
  
  def vgg_layers(self, layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable=False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    
    model = tf.keras.Model([vgg.input], outputs)
    return model
  
  def call(self, inputs):
    "Expects float input in [0,1]" # TODO: lookup this line
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                     outputs[self.num_style_layers:])
    
    style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
    
    content_dict = {content_name:value for content_name, value in zip(
        self.content_layers, content_outputs
    )}
    
    style_dict = {style_name:value for style_name, value in zip(
        self.style_layers, style_outputs
    )}
    
    return {'content':content_dict, 'style':style_dict}    
  
  def gram_matrix(self, input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations) # why the parenthesis?

def trim_filename(filename):
  assert(type(filename) is str)
  # Trim the extension from filename
  filename = filename[:-(filename[::-1].index('.')+1)]
  # Strip all non-alpha characters
  filename = ''.join([letter for letter in filename if letter.isalpha()])
  return filename

if __name__ == '__main__':
  
  output_path = './output/'
  style_images = []
  content_images = []
  settings = [
      {'style_weight':1e-2, 'content_weight':1e4, 'total_variation_weight':30, 'epochs':10},
      {'style_weight':1e-2, 'content_weight':1e4, 'total_variation_weight':30, 'epochs':20},
      {'style_weight':1e-2, 'content_weight':1e4, 'total_variation_weight':30, 'epochs':30},
      {'style_weight':1e-2, 'content_weight':1e4, 'total_variation_weight':60, 'epochs':20},
      {'style_weight':1e-2, 'content_weight':1e4, 'total_variation_weight':60, 'epochs':30}
  ]
  
  directory = os.listdir('./')
  
  # this section needs serious refactoring:
  
  if directory.count('style'):
    for img in os.listdir('./style'):
      style_images.append('./style/'+img)
  else:
    raise Exception('No ./style directory exists')
    
  if directory.count('content'):
    for img in os.listdir('./content'):
      content_images.append('./content/'+img)
  else:
    raise Exception('No . /content directory exists')
  
  # Maybe change these to assert calls?
  # TODO change this to search output_path
  assert(os.listdir().count('output'))
  
  for setting_index,setting in enumerate(settings):
    print(setting)
    
    for style_image in style_images:
      print(style_image)
      
      for content_image in content_images:
        print(content_image)
        print('style_processor call')
        stylizer = style_processor(content_image, style_image, **setting)
        
        output, *_ = stylizer.process()
        
        output.save(output_path
                    +trim_filename(content_image)
                    +'_as_'+trim_filename(style_image)
                    +'_setting-'+str(setting_index)
                    +'.jpeg')
        
        #assert(False), "ENDING EARLY"
  
  
  
  
    
    
    
    