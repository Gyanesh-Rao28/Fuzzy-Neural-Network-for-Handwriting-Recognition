import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Display 10 random samples from the training set
def display_samples(x_train, y_train):
    plt.figure(figsize=(10, 5))
    for i in range(30):
        plt.subplot(6, 5, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

display_samples(x_train, y_train)
