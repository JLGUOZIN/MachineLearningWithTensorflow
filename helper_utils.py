import tensorflow as tf
import matplotlib.pyplot as plt
from tf_keras.applications.efficientnet import preprocess_input

def pred_and_plot_jeff(model, class_names, filename, img_shape=224):
  image_tensor = tf.io.read_file(filename)
  # Decode the image and ensure 3 color channels (RGB)
  image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
  # Resize the image to the target shape
  image_tensor = tf.image.resize(image_tensor, [img_shape, img_shape])
  # Preprocess the image for EfficientNet
  image_tensor = preprocess_input(image_tensor)

  # Expand dimensions to match model input (batch size, height, width, channels)
  test_img_batch = tf.expand_dims(image_tensor, axis=0)

  # Make predictions
  predictions = model.predict(test_img_batch)

  # Determine the predicted class
  predicted_class = class_names[tf.argmax(predictions[0])]

  # Print and plot the result
  print(f"Predicted class: {predicted_class}")
  plt.imshow(tf.image.decode_jpeg(tf.io.read_file(filename)))
  plt.title(f"Prediction: {predicted_class}")
  plt.axis(False)
  plt.show()


# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    img = img/255.
  else:
    img = preprocess_input(img)
  return img

def pred_and_plot(model, filename, class_names, img_shape, scale):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename, img_shape, scale)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[tf.argmax(pred[0])] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

  # Prepare image for display
  if not scale:
    img_for_display = (img + 1) / 2  # Scale [-1, 1] to [0, 1] (if preprocess_input was used)
  else:
    img_for_display = img  # Already in [0, 1] range

  img_for_display = tf.clip_by_value(img_for_display, clip_value_min=0.0, clip_value_max=1.0)

  # Plot the image and predicted class
  plt.imshow(img_for_display)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)
  plt.show()

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();