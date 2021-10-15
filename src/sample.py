import tensorflow as tf
import numpy as np
from PIL import Image

print(tf.__version__)
# 2.1.0

tf.random.set_seed(0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

Image.fromarray(x_train[0]).resize((256, 256)).save('tmp/mnist_sample_resize.png')

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten_layer'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
], name='my_model')

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
             tf.keras.callbacks.ModelCheckpoint(
                 'tmp/mnist_sequential_{epoch:03d}_{val_loss:.4f}.ckpt',
                 save_best_only=True
             )]

history = model.fit(x_train, y_train, batch_size=128, epochs=20,
                    validation_split=0.2, callbacks=callbacks)

print(type(history))
# <class 'tensorflow.python.keras.callbacks.History'>

print(type(history.history))
# <class 'dict'>

print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

print(history.history['accuracy'])
# [0.87514585, 0.93927085, 0.9539792, 0.96164584, 0.967125, 0.9710417, 0.9741875, 0.9771875, 0.97866666, 0.98075, 0.98310417, 0.98441666]


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("test_loss: ", test_loss)
# 0.07156232252293267

print("test_acc: ", test_acc)
# 0.9786

# model.save('tmp/my_model.h5')
# tf.saved_model.save(model, 'tmp/saved_model')
model.save('tmp/my_model_for_saved_model/')

# new_model = tf.keras.models.load_model('tmp/my_model.h5')
# new_model = tf.saved_model.load("tmp/saved_model")
new_model = tf.keras.models.load_model("tmp/my_model_for_saved_model")
new_model.summary()
# Model: "my_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# flatten_layer (Flatten)      (None, 784)               0
# _________________________________________________________________
# dense (Dense)                (None, 128)               100480
# _________________________________________________________________
# dropout (Dropout)            (None, 128)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1290
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
# _________________________________________________________________


img = np.array(Image.open('tmp/mnist_sample_resize.png').resize((28, 28))) / 255
print(img.shape)
# (28, 28)

print(new_model.evaluate(x_test, y_test, verbose=0))
# [0.07156232252293267, 0.9786]

img_expand = img[tf.newaxis]

print(new_model.predict(img_expand))
# [[6.8237398e-09 1.0004978e-08 4.0168429e-06 4.5704491e-02 2.8772252e-14
#   9.5429057e-01 5.5912805e-12 3.1738683e-08 6.9545142e-10 9.2607473e-07]]
