# %% Load Packages 
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import os
import PIL
import time
from IPython import display
from numpy.random import randn
from numpy.random import randint

# %% Load Dataset (Fashion-MNIST Clothing Photograph Dataset)
# load the images into memory
(trainX, trainy), (testX, testy) = tf.keras.datasets.fashion_mnist.load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

# %% Sample Image Visualization
# plot images from the training dataset
for i in range(100):
    # define subplot
    plt.subplot(10, 10, 1 + i)
    # turn off axis
    plt.axis('off')
    # plot raw pixel data
    plt.imshow(trainX[i], cmap='gray_r')
plt.show()

# %% Resize (28 X 28), Image scaling (-1 ~ 1)
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1).astype('float32')
trainX = (trainX - 127.5) / 127.5 # 이미지를 [-1, 1]로 정규화합니다.



# %% Discriminator
class Discriminator(tf.keras.Model):
  def __init__(self, n_classes = 10, embed_dim = 50, input_shape=(28,28,1)):
      super(Discriminator, self).__init__()
      ## y_label layers
      # embedding for categorical input
      self.y_embedding = layers.Embedding(n_classes, embed_dim)
      # scale up to image dimensions with linear activation
      n_nodes = input_shape[0] * input_shape[1]
      self.y_dense = layers.Dense(n_nodes)
      # reshape to additional channel
      self.y_reshape = layers.Reshape(input_shape)
      # concat label as a channel
      self.merge = layers.Concatenate()
      # downsample
      self.input_conv1 = layers.Conv2D(128, (3,3), strides=(2,2), padding='same')
      self.input_leaky_relu1 = layers.LeakyReLU(alpha=0.2)
      # downsample
      self.input_conv2 = layers.Conv2D(128, (3,3), strides=(2,2), padding='same')
      self.input_leaky_relu2 = layers.LeakyReLU(alpha=0.2)
      # flatten feature maps
      self.input_flat = layers.Flatten()
      # dropout
      self.input_dropout = layers.Dropout(0.4)
      # output
      self.input_dense = layers.Dense(1, activation='sigmoid')

  def call(self, in_x, in_y):
      ## Embedding input labels
      embed_y = self.y_embedding(in_y)
      embed_y = self.y_dense(embed_y)
      embed_y = self.y_reshape(embed_y)
      ## Concatnate
      merged_input = self.merge([embed_y, in_x])
      ## Discriminator
      output = self.input_conv1(merged_input)
      output = self.input_leaky_relu1(output)
      output = self.input_conv2(output)
      output = self.input_leaky_relu2(output)
      output = self.input_flat(output)
      output = self.input_dropout(output)
      output = self.input_dense(output)
      return output

# %%
# Generator
## "padding=same": o = ceil(i/s), where o = output size, i = input size, s = stride
## 즉 stride=2인 경우 output은 절반 사이즈.
class Generator(tf.keras.Model):
  def __init__(self, n_classes = 10, embed_dim = 50):
      super(Generator, self).__init__()
      ## y_label layers
      # embedding for categorical input
      self.y_embedding = layers.Embedding(n_classes, embed_dim)
      # linear multiplication
      self.y_dense = layers.Dense(49)
      # reshape to additional channel
      self.y_reshape = layers.Reshape((7,7,1))
      # foundation for 7x7 image
      n_nodes = 128 * 7 * 7
      self.input_dense = layers.Dense(n_nodes)
      self.input_leaky_relu1 = layers.LeakyReLU(alpha=0.2)
      self.input_reshape = layers.Reshape((7, 7, 128))
      # merge image gen and label input
      self.merge = layers.Concatenate()
      # upsample to 14x14
      self.input_conv1 = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
      self.input_leaky_relu2 = layers.LeakyReLU(alpha=0.2)
      # upsample to 28x28
      self.input_conv2 = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
      self.input_leaky_relu3 = layers.LeakyReLU(alpha=0.2)
      # output
      self.input_conv3 = layers.Conv2D(1, (7,7), activation='tanh', padding='same')

  def call(self, in_lat, in_y):
      ## Embedding input labels
      embed_y = self.y_embedding(in_y)
      embed_y = self.y_dense(embed_y)
      embed_y = self.y_reshape(embed_y)
      # foundation for 7x7 image
      latent = self.input_dense(in_lat)
      latent = self.input_leaky_relu1(latent)
      latent = self.input_reshape(latent)
      ## Concatnate
      merged_input = self.merge([embed_y, latent])
      ## Discriminator
      output = self.input_conv1(merged_input)
      output = self.input_leaky_relu2(output)
      output = self.input_conv2(output)
      output = self.input_leaky_relu3(output)
      output = self.input_conv3(output)
      return output



# %% Loss 정의
## from_logits: softmax를 지나지 않은 값
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# %% Optimizer 정의
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# %% Generate samples
 
# # select real samples
def generate_real_samples(images, labels, n_samples):
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	image_batch, label_batch = images[ix], labels[ix]
	return image_batch, label_batch
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return z_input, labels

# %%


# %% Define Training loops
## noise_dim: Generator는 처음 시작에 Noise (Random vector)로 부터 이미지를 생성함.
## num_examples_to_generate: 생성할 이미지의 갯수
EPOCHS = 30
noise_dim = 100
BATCH_SIZE = 128
num_examples_to_generate = 16
# 이 시드를 시간이 지나도 재활용하겠습니다. 
# (GIF 애니메이션에서 진전 내용을 시각화하는데 쉽기 때문입니다.) 
seed = tf.random.normal([num_examples_to_generate, noise_dim])
seed_label = randint(0, 10, num_examples_to_generate)
# %% Train step
# `tf.function` 자체가 함수를 Decorate 함.
# 이 데코레이터는 함수를 "컴파일"합니다. (Tensor 연산 속도 향상)
@tf.function
def train_step(images, images_label):
    noise, noise_label = generate_latent_points(noise_dim, BATCH_SIZE)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = Generator_model(noise, noise_label, training=True)

      real_output = Discriminator_model(images, images_label, training=True)
      fake_output = Discriminator_model(generated_images, noise_label, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, Generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, Discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, Generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, Discriminator_model.trainable_variables))

# %% Train

def train(images, images_label):
  for epoch in range(EPOCHS):
    start = time.time()
    bat_per_epoch = int(images.shape[0] / BATCH_SIZE)
    for i in range(bat_per_epoch):
        image_batch, label_batch = generate_real_samples(images, images_label, BATCH_SIZE)
        train_step(image_batch, label_batch)


    # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # 마지막 에포크가 끝난 후 생성합니다.
  display.clear_output(wait=True)
  generate_and_save_images(Generator_model, EPOCHS, seed, seed_label)

# %% Create Image and Save it.
def generate_and_save_images(model, epoch, test_input, test_y):
  # `training`이 False로 맞춰진 것을 주목하세요.
  # 이렇게 하면 (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됩니다. 
  predictions = model(test_input, test_y, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# %% Create the models
Discriminator_model = Discriminator()
Generator_model = Generator()

# %% Train the models
train(trainX, trainy)

# %%
