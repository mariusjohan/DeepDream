# Import dependencies
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3 # Works fine
from tensorflow.keras.preprocessing import image # Works fine
import numpy as np
from PIL import Image
import time

# Loads the pretrained inception model by Google
model = InceptionV3(
    include_top = False,
    weights = 'imagenet'
)

# Function to open images and load them into a numpy array
def open_img(img_path, max_dim=None):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        if max_dim:
            img = img.thumbnail((max_dim, max_dim))
        img_arr = np.asarray(img)
    return img_arr

# Converts the image back into a "real" image
def deprocess_img(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

# Save numpy array to a image file
def save_image(img, img_path):
    img.save(img_path)

# Gets the layers for deep dream
names = ['mixed3','mixed5']
layers = [model.get_layer(n).output for n in names]

# Initializes the deep dream model
deep_dream_model = tf.keras.Model(
    inputs=model.input, 
    outputs=layers
)

# Function to calculate loss
def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)

    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for a in layer_activations:
        loss = tf.reduce_mean(a)
        losses.append(loss)

    return tf.reduce_sum(losses)

# Custom DeepDream module
# The purpose is to produce a gradient ascent (not descent)
class DeepDream(tf.Module):

    def __init__(self, model):
        # Model is the custom keras model with the inception input, mixed3 & mixed5 layers
        self.model = model

    @tf.function(
        input_signature = (
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32)
        )
    )
    def __call__(self, img, epochs, epoch_size):
        loss = tf.constant(0.0)

        # Computes gradient ascent
        for i in range(epochs):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)

            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            img += gradients * epoch_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img

# Initializes the custom deep dream module
deepdream = DeepDream(deep_dream_model)

# Runs the DeepDream module one time
def run(img, epochs, step_size):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)

    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = epochs
    step = 0

    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)

        steps_remaining -= 1
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))

        print(f'Steps left {steps_remaining}, loss {loss:.4}')

        # Save results every 20 steps
        if steps_remaining % 5 == 0:
            print('Saved image')
            saving_img = deprocess_img(img)
            saving_img = convert_tensor_to_numpy(saving_img)
            save_image(saving_img, f'output-image-{steps_remaining}.png')


    return deprocess_img(img)

# Runs the DeepDream multiple times (5) 
# and with octaves
def run_advanced(img, epochs, step_size, octave_scale):
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)

    t = time.time()
    for i in range(-2, 3):
        new_shape = tf.cast(float_base_shape*(octave_scale**i), tf.int32)
        img = tf.image.resize(img, new_shape).numpy()
        img = run(img, epochs, step_size)

        # Prints how much time it took
        print(time.time() - t)
        t = time.time()

    return img

def convert_tensor_to_numpy(img):
    img = tf.keras.preprocessing.image.array_to_img(img)
    return img

if __name__ == "__main__":
    img = open_img('van-gogh.jpg')
    img = run(img, 100, 0.01)
    img = convert_tensor_to_numpy(img)
    # img = run_advanced(img, 100, 0.01, 1.3)
    save_image(img, 'output-image.png')