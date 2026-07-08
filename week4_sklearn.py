import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.neural_network import MLPClassifier

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
data = mnist.data / 255.0
labels = mnist.target

train_images = data[:5000]
train_labels = labels[:5000]
test_images = data[5000:5100]
test_labels = labels[5000:5100]

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_images, train_labels)
predictions = model.predict(test_images)

accuracy = np.mean(predictions == test_labels)
print("sklearn kNN accuracy: ", accuracy)

print("\nTraining Logistic Regression...")
start = time.time()
model2 = LogisticRegression(max_iter=1000)
model2.fit(train_images, train_labels)
predictions2 = model2.predict(test_images)
accuracy2 = np.mean(predictions2 == test_labels)
print("Logistic Regression accuracy: ", accuracy2)
print("Time: ", round(time.time() - start, 1), "sec")

print("\nTraining Neural Network...")
start = time.time()
model3 = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20)
model3.fit(train_images, train_labels)
predictions3 = model3.predict(test_images)
accuracy3 = np.mean(predictions3 == test_labels)
print("Neural Network accuracy: ", accuracy3)
print("Time: ", round(time.time() - start, 1), "sec")