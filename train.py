import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, save_model
from keras.layers import Dense, InputLayer
from keras.optimizers.legacy import RMSprop
import matplotlib.pyplot as plt

# Prepare data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, (-1, 784))
X_test = np.reshape(X_test, (-1, 784))
print(X_train.shape, X_test.shape)

# Show a random training image
img_id = np.random.randint(len(X_train))
img = np.reshape(X_train[img_id], (28, 28))
plt.imshow(img)
# plt.show()

# Create the model
model = Sequential()
model.add(InputLayer((784,)))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile, train and save the model
optimizer = RMSprop(learning_rate=2e-4, decay=1e-6)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
)
save_model(model, "model.keras", save_format="keras")
