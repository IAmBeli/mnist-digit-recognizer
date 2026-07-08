import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

print("Loading data... (first run downloads ~15 MB, wait a minute)")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

print("Done!")
print("Data shape:", mnist.data.shape)
print("Labels shape:", mnist.target.shape)
print("First 10 answers:", mnist.target[:10])

first_image = mnist.data[0].reshape(28, 28)
plt.imshow(first_image, cmap="gray")
plt.title("Label: " + mnist.target[0])
plt.show()