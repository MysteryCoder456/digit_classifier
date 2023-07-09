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
    random_img = X_test[idx]
    pred = model.predict(np.array([random_img]))
    pred_num = np.argmax(pred)

    plt.imshow(random_img)
    plt.title(f"Prediction = {pred_num}")
    plt.xlabel(f"Actual = {y_test[idx]}")

    plt.show()
