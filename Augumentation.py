# %% Load Packages 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
# %% 이 튜토리얼에서는 tf_flowers 데이터세트를 사용합니다.
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
# %% Make Sample
num_classes = metadata.features['label'].num_classes
print(num_classes)

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))

# %% Comparison Original and Augmented images
def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)

# %% 이미지 뒤집기 (이미지를 수직 또는 수평으로 뒤집습니다.)
flipped = tf.image.flip_left_right(image)
visualize(image, flipped)

flipped = tf.image.flip_up_down(image)
visualize(image, flipped)
# %% 이미지를 회색조로 만들기
grayscaled = tf.image.rgb_to_grayscale(image)
visualize(image, tf.squeeze(grayscaled))
_ = plt.colorbar()

# %% 이미지 포화시키기
saturated = tf.image.adjust_saturation(image, 3)
visualize(image, saturated)

# %% 이미지 밝기 변경하기
bright = tf.image.adjust_brightness(image, 0.4)
visualize(image, bright)

# %% 이미지 중앙 자르기
cropped = tf.image.central_crop(image, central_fraction=0.5)
visualize(image,cropped)

# %% 이미지 회전하기
rotated = tf.image.rot90(image)
visualize(image, rotated)

# %% Dataset에 Augumentation 적용하기.
def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image, label

def augment(image,label):
  image, label = resize_and_rescale(image, label)
  # Add 6 pixels of padding
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6) 
   # Random crop back to the original size
  image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
  image = tf.clip_by_value(image, 0, 1)
  return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 32
IMG_SIZE = 180

train_ds = (
    train_ds
    .shuffle(1000)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)
# %%
train_ds
# %%
