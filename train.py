from datetime import datetime
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers.legacy import RMSprop
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
model.add(Flatten(input_shape=(28, 28)))

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))

# Output Layer
model.add(Dense(10, activation="softmax"))

# Compile, train and save the model
optimizer = RMSprop(learning_rate=15e-5, decay=1e-6)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[tensorboard_callback],
)
save_model(model, "model.keras", save_format="keras")
