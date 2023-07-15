from datetime import datetime
from keras.src.optimizers import SGD
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, save_model
from keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    InputLayer,
    MaxPooling2D,
    AveragePooling2D,
    Reshape,
)
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# Prepare data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Show a random training image
img_id = np.random.randint(len(X_train))
img = X_train[img_id]
plt.imshow(img)
# plt.show()

# Initializing tensorboard
log_dir = "logs/" + datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Create the model
model = Sequential()

# Input Layer
model.add(InputLayer(input_shape=(28, 28)))
model.add(Reshape((28, 28, 1)))

# Convolutional Layers

model.add(Conv2D(58, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D((3, 3)))
model.add(Dropout(0.4))

model.add(Conv2D(28, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(AveragePooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(28, (2, 2)))
model.add(Activation("relu"))
model.add(Dropout(0.35))

# Neural Layers

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))

# Output Layer
model.add(Dense(10, activation="softmax"))

# Compile, train and save the model
# rms_prop = RMSprop(learning_rate=3e-4, epsilon=1e-6)
sgd = SGD(learning_rate=7e-4, momentum=0.9)

model.compile(
    optimizer=sgd,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    callbacks=[tensorboard_callback],
)
save_model(model, "model.keras", save_format="keras")
