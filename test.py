import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt

# Load model
model: Sequential = load_model("model.keras", compile=False)  # type: ignore
(_, _), (X_test, y_test) = mnist.load_data()

# Test 10 random images
for _ in range(10):
    idx = np.random.randint(len(X_test))
    random_img = np.reshape(X_test[idx : idx + 1], (-1, 784))
    pred = model.predict(random_img)
    pred_num = np.argmax(pred)

    plt.imshow(X_test[idx])
    plt.title(f"Prediction = {pred_num}")
    plt.xlabel(f"Actual = {y_test[idx]}")

    plt.show()
