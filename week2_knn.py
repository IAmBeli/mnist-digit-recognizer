import numpy as np
from sklearn.datasets import fetch_openml
from collections import Counter

def distance(image_a, image_b):
    diff = image_a - image_b
    squarepower = diff**2
    total = np.sum(squarepower)
    result = np.sqrt(total)
    return result
def predict(test_image, train_images, train_labels, k=5):
    distances = [distance(test_image, train_img) for train_img in train_images]
    nearest_ids = np.argsort(distances)[:k]
    nearest_labels = [train_labels[i] for i in nearest_ids]
    winner = Counter(nearest_labels).most_common(1)[0][0]
    return winner

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
data = mnist.data
labels = mnist.target

train_images = data[:5000]
train_labels = labels[:5000]

test_images = data[5000:5100]
test_labels = labels[5000:5100]

correct = 0
for i in range(100):
    prediction = predict(test_images[i], train_images, train_labels, k=5)
    if prediction == test_labels[i]:
        correct += 1
    print(f"Image {i}: true={test_labels[i]}, predicted={prediction}")

accuracy = correct / 100
print("Accuracy: ", accuracy)