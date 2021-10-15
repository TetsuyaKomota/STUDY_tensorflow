import tensorflow as tf
import numpy as np
from PIL import Image

print(tf.__version__)
# 2.1.0

tf.random.set_seed(0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        # self.optimizer = tf.keras.optimizers.Adam()
        self.my_metrics = {
                "loss"     : tf.keras.metrics.Mean(name='train_loss'),
                "accuracy" : tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'),
            }


    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.my_metrics["loss"].update_state(loss)
        self.my_metrics["accuracy"].update_state(labels, predictions)
        return {name: metric.result() for name, metric in self.my_metrics.items()}

    def test_step(self, data):
        images, labels = data
        predictions = self(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.my_metrics["loss"].update_state(t_loss)
        self.my_metrics["accuracy"].update_state(labels, predictions)
        return {name: metric.result() for name, metric in self.my_metrics.items()}


model = MyModel()

model.compile(optimizer="Adam")

callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('tmp/expert.model_{epoch:03d}_{val_loss:.4f}.ckpt',save_best_only=True), 
        ]

model.fit(train_ds, epochs=20, validation_data=test_ds, callbacks=callbacks)

model.save("tmp/expert.model")

new_model = tf.keras.models.load_model("tmp/expert.model")

new_model.summary()


img = np.array(Image.open('tmp/mnist_sample_resize.png').resize((28, 28))) / 255
img_expand = img[tf.newaxis]
print(new_model.predict(img_expand))
