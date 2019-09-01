
# The source is "https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras"

#############################################################################################################
###################################       Pruning         ###################################################
#############################################################################################################
import tensorboard
import tensorflow as tf
#tf.enable_eager_execution()
import tempfile
import zipfile
import os
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np
l = tf.keras.layers

batch_size = 128
num_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


#############################################################################################################
###################################       1.Pruning by layers        #########################################
#############################################################################################################
num_train_samples = x_train.shape[0]
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
print('End step: ' + str(end_step))
pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,final_sparsity=0.90,
                                                               begin_step=2000,end_step=end_step, frequency=100)}
pruned_model = tf.keras.Sequential([
    sparsity.prune_low_magnitude(l.Conv2D(32, 5, padding='same', activation='relu'), input_shape=input_shape,
                                 **pruning_params),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'), l.BatchNormalization(),
    sparsity.prune_low_magnitude(l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.Flatten(),
    sparsity.prune_low_magnitude(l.Dense(1024, activation='relu'), **pruning_params),
    l.Dropout(0.4),
    sparsity.prune_low_magnitude(l.Dense(num_classes, activation='softmax'), **pruning_params)
])
pruned_model.summary()

logdir = tempfile.mkdtemp()
print('Writing training logs to ' + logdir)
# Train the model
# Start pruning from step 2000 when accuracy >98%
pruned_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

# Add a pruning step callback to peg the pruning step to the optimizer's
# step. Also add a callback to add pruning summaries to tensorboard
callbacks = [sparsity.UpdatePruningStep(), sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)]
pruned_model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=1, callbacks=callbacks,
                 validation_data=(x_test, y_test))
score = pruned_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save and restore the pruned model
# Continue training for two epochs:
_, checkpoint_file = tempfile.mkstemp('.h5')
print('Saving pruned model to: ', checkpoint_file)
# saved_model() sets include_optimizer to True by default. Spelling it out here
# to highlight.
tf.keras.models.save_model(pruned_model, checkpoint_file, include_optimizer=True)
with sparsity.prune_scope():
    restored_model = tf.keras.models.load_model(checkpoint_file)
restored_model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=2,
                   verbose=1,
                   callbacks=callbacks,
                   validation_data=(x_test, y_test))
score = restored_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#############################################################################################################
#####          2.Strip the pruning wrappers from the pruned model before export for serving              ######
#############################################################################################################
# Strip the pruning wrappers from the pruned model before export for serving
# Before exporting a serving model,
# you'd need to call the strip_pruning API to strip the pruning wrappers from the model,
# as it's only needed for training.
final_model = sparsity.strip_pruning(pruned_model)
final_model.summary()
_, pruned_keras_file = tempfile.mkstemp('.h5')
print('Saving pruned model to: ', pruned_keras_file)
# No need to save the optimizer with the graph for serving.
tf.keras.models.save_model(final_model, pruned_keras_file, include_optimizer=False)
# Zip the weights
_, zip2 = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip2, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(pruned_keras_file)
print("Size of the pruned model before compression: %.2f Mb" % (os.path.getsize(pruned_keras_file) / float(2**20)))
print("Size of the pruned model after compression: %.2f Mb" % (os.path.getsize(zip2) / float(2**20)))


#############################################################################################################
############################          3.Convert to TensorFlow Lite              #############################
#############################################################################################################
tflite_model_file = '/tmp/sparse_mnist.tflite'
#converter = tf.lite.TFLiteConverter.from_keras_model_file(pruned_keras_file)
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(pruned_keras_file)
tflite_model = converter.convert()
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)
# Zipping the weights
_, zip_tflite = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip_tflite, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_model_file)
print("Size of the tflite model before compression: %.2f Mb" % (os.path.getsize(tflite_model_file) / float(2**20)))
print("Size of the tflite model after compression: %.2f Mb" % (os.path.getsize(zip_tflite) / float(2**20)))


#############################################################################################################
##################         4.Evaluate the accuracy of the TensorFlow Lite model          ####################
#############################################################################################################
#
import numpy as np

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def eval_model(interpreter, x_test, y_test):
  total_seen = 0
  num_correct = 0
  for img, label in zip(x_test, y_test):
    inp = img.reshape((1, 28, 28, 1))
    total_seen += 1
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    if np.argmax(predictions) == np.argmax(label):
      num_correct += 1

    if total_seen % 1000 == 0:
        print("Accuracy after %i images: %f" %
              (total_seen, float(num_correct) / float(total_seen)))

  return float(num_correct) / float(total_seen)
print(eval_model(interpreter, x_test, y_test))


#############################################################################################################
###################         5.Post-training quantize the TensorFlow Lite model          #####################
#############################################################################################################
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(pruned_keras_file)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
tflite_quant_model_file = '/tmp/sparse_mnist_quant.tflite'
with open(tflite_quant_model_file, 'wb') as f:
    f.write(tflite_quant_model)
_, zip_tflite = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip_tflite, 'w', compression=zipfile.ZIP_DEFLATED) as f:
  f.write(tflite_quant_model_file)
print("Size of the tflite model before compression: %.2f Mb" % (os.path.getsize(tflite_quant_model_file) / float(2**20)))
print("Size of the tflite model after compression: %.2f Mb" % (os.path.getsize(zip_tflite) / float(2**20)))
