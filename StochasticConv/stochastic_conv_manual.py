import numpy as np
import matplotlib.pyplot as plt
import struct

np.random.seed(1234)

# ----------------------------
# Load MNIST images and labels
# ----------------------------
def load_data(size=True):
    def load_images(filename):
        with open(filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            return image_data.reshape(num_images, rows, cols)

    def load_labels(filename):
        with open(filename, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    def one_hot_encode(labels, num_classes=10):
        encoded = np.zeros((len(labels), num_classes), dtype=int)
        encoded[np.arange(len(labels)), labels] = 1
        return encoded

    train_images = load_images(r'.data\MNIST\raw\train-images-idx3-ubyte')
    train_labels = load_labels(r'.data\MNIST\raw\train-labels-idx1-ubyte')
    test_images = load_images(r'.data\MNIST\raw\t10k-images-idx3-ubyte')
    test_labels = load_labels(r'.data\MNIST\raw\t10k-labels-idx1-ubyte')

    if size:
        print(f"Train Images Shape: {train_images.shape}")
        print(f"Test Images Shape: {test_images.shape}")

    X_train = train_images.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = test_images.reshape(-1, 784).astype(np.float64) / 255.0
    y_train = one_hot_encode(train_labels)
    y_test = one_hot_encode(test_labels)

    return X_train, y_train, X_test, y_test

# ----------------------------
# Manual 2D Convolution (no padding change)
# ----------------------------
def convolve2d_manual(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros_like(image, dtype=float)

    for i in range(H):
        for j in range(W):
            region = padded[i:i + kH, j:j + kW]
            result[i, j] = np.sum(region * kernel)

    return result

# ----------------------------
# Single stochastic kernel with Monte Carlo noise
# ----------------------------
class StochasticKernel:
    def __init__(self, size=3, T=100):
        self.size = size
        self.T = T
        self.threshold = np.random.uniform(-0.1, 0.1, size=(size, size))
        self.sigma = np.random.uniform(0.8, 1.2, size=(size, size))
        self.kernel = np.random.uniform(-1, 1, size=(size, size))

    def SRA_sigmoid(self, threshold, kernel, sigma):
        noise = np.random.normal(loc=0, scale=sigma, size=self.T)
        noisy_values = 2 * kernel + noise
        return 2 * (noisy_values > threshold).mean() - 1

    def stochastic_forward(self):
        new_kernel = np.zeros_like(self.kernel)
        for i in range(self.size):
            for j in range(self.size):
                new_kernel[i, j] = self.SRA_sigmoid(
                    self.threshold[i, j], self.kernel[i, j], self.sigma[i, j]
                )
        self.kernel = new_kernel
        return self.kernel

# ----------------------------
# Multi-kernel stochastic convolution layer
# ----------------------------
class StochasticConvLayer:
    def __init__(self, num_kernels, kernel_size=3, T=100):
        self.kernels = [StochasticKernel(kernel_size, T) for _ in range(num_kernels)]

    def forward(self, input_image):
        outputs = []
        for kernel in self.kernels:
            # Uncomment below to apply stochastic update each pass
            # kernel.stochastic_forward()
            output = convolve2d_manual(input_image, kernel.kernel)
            outputs.append(output)
        return np.stack(outputs, axis=0)

# ----------------------------
# Grid plot of original image + output feature maps
# ----------------------------
def plot_kernels():
    num_maps = output_map.shape[0]
    grid_cols = int(np.ceil(np.sqrt(num_maps + 1)))
    grid_rows = int(np.ceil((num_maps + 1) / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 12))
    axes = axes.flatten()

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original", fontsize=8)
    axes[0].axis('off')

    for i in range(num_maps):
        axes[i + 1].imshow(output_map[i], cmap='gray')
        axes[i + 1].set_title(f"Kernel {i + 1}", fontsize=8)
        axes[i + 1].axis('off')

    for j in range(num_maps + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# ----------------------------
# Run demo: load image → apply conv layer → plot
# ----------------------------
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data(size=False)
    image = X_train[0].reshape(28, 28)

    stochastic_conv_layer = StochasticConvLayer(num_kernels=35, kernel_size=3, T=100)
    output_map = stochastic_conv_layer.forward(image)

    plot_kernels()
