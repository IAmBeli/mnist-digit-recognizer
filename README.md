# MNIST Digit Recognizer

A from-scratch machine learning project that recognizes handwritten digits (0–9) from the MNIST dataset. Built as a hands-on learning journey — starting with a self-implemented algorithm, accelerating it with C++, and finishing with industry-standard tools.

## Overview

The goal of this project was not just to classify digits, but to **understand how machine learning works from the ground up**. Instead of calling a library first, I implemented the k-Nearest Neighbors (kNN) algorithm by hand, then progressively introduced professional tools and compared them.

Each image is a 28×28 grayscale picture, flattened into 784 pixel values. The models learn to map these pixels to the correct digit.

## What's Inside

| File | Description |
|------|-------------|
| `week1_data.py` | Loading and visualizing the MNIST dataset with numpy and matplotlib |
| `week2_knn.py` | k-Nearest Neighbors implemented from scratch (Euclidean distance, voting) |
| `knn_core.cpp` | C++ implementation of the distance function, bound to Python via pybind11 |
| `week4_sklearn.py` | Comparison of scikit-learn models: kNN, Logistic Regression, Neural Network |

## Results

Models evaluated on held-out test images (accuracy on 100 samples):

| Model | Accuracy | Notes |
|-------|----------|-------|
| kNN (from scratch) | 96% | Hand-written Euclidean distance + voting |
| kNN (scikit-learn) | 96% | Matches the from-scratch version |
| Logistic Regression | 85% | Very fast training (~0.2s), learns rules instead of storing images |
| Neural Network (MLP) | 97% | Best result; one hidden layer of 128 neurons |

## Python vs C++ Performance

The distance calculation was rewritten in C++ and compared against pure Python:

| Version | Time (50 images) | Speedup |
|---------|------------------|---------|
| Pure Python (manual loop) | ~19.8s | 1× (baseline) |
| C++ (via pybind11) | ~0.5s | **~37× faster** |
| Python + numpy | ~0.6s | ~33× faster |

**Key takeaway:** C++ dramatically speeds up manual pixel-by-pixel loops, but numpy (written in C internally) is already nearly as fast — so C++ is worth it only for hot spots numpy doesn't cover.

## Tech Stack

- **Python** — main language
- **C++** — performance-critical distance function
- **pybind11** — bridge between Python and C++
- **scikit-learn** — production ML models
- **numpy / matplotlib** — data handling and visualization

## How to Run

Install dependencies:

```bash
pip3 install numpy matplotlib scikit-learn pybind11
```

Run the data loader:

```bash
python3 week1_data.py
```

Run the from-scratch kNN:

```bash
python3 week2_knn.py
```

Compile the C++ module (requires a C++ compiler):

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC -undefined dynamic_lookup \
    $(python3 -m pybind11 --includes) knn_core.cpp \
    -o knn_core$(python3-config --extension-suffix)
```

Compare scikit-learn models:

```bash
python3 week4_sklearn.py
```

## What I Learned

- How the kNN algorithm works internally (distance metrics, nearest-neighbor voting)
- Writing and compiling C++ extensions for Python with pybind11
- The difference between "lazy" (kNN) and "eager" (Logistic Regression, Neural Networks) learning algorithms
- When optimization actually matters — and when numpy is already fast enough
- Measuring model quality with accuracy and comparing trade-offs between speed and precision

---

*Built as a project-based introduction to machine learning and AI/ML engineering.*