import numpy as np
import struct
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

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

    def one_hot_encode_to_array(labels, num_classes=10):
        encoded = np.zeros((len(labels), num_classes), dtype=int)
        encoded[np.arange(len(labels)), labels] = 1
        return encoded

    train_images = load_images(r'.data\MNIST\raw\train-images-idx3-ubyte')
    train_labels = load_labels(r'.data\MNIST\raw\train-labels-idx1-ubyte')
    test_images = load_images(r'.data\MNIST\raw\t10k-images-idx3-ubyte')
    test_labels = load_labels(r'.data\MNIST\raw\t10k-labels-idx1-ubyte')

    if size:
        print(f"Train Images Shape: {train_images.shape}")
        print(f"Train Labels Shape: {train_labels.shape}")
        print(f"Test Images Shape: {test_images.shape}")
        print(f"Test Labels Shape: {test_labels.shape}")

    X_train = train_images.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = test_images.reshape(-1, 784).astype(np.float64) / 255.0

    y_train = one_hot_encode_to_array(train_labels)
    y_test = one_hot_encode_to_array(test_labels)

    return X_train, y_train, X_test, y_test

# ----------------------------
# Stochastic kernel layer with resampled weights
# ----------------------------
class StochasticKernelLayer:
    class StochasticKernel:
        def __init__(self, kernel_size, initial_mu=0.0, initial_sigma=1.0):
            self.kernel_size = kernel_size
            self.threshold = initial_mu
            self.sigma = initial_sigma
            self.weights = np.random.uniform(-1, 1, size=(kernel_size, kernel_size))

        def sample_kernel(self):
            return np.random.normal(loc=self.threshold, scale=self.sigma,
                                    size=(self.kernel_size, self.kernel_size))

    def __init__(self, num_kernels, kernel_size):
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.initial_mu_list = np.random.uniform(-0.1, 0.1, size=num_kernels)
        self.initial_sigma_list = np.random.uniform(0.8, 1.2, size=num_kernels)
        self.kernels = [
            self.StochasticKernel(kernel_size, self.initial_mu_list[i], self.initial_sigma_list[i])
            for i in range(num_kernels)
        ]

    def sample_kernels(self):
        sampled_kernels = []
        new_kernels = []

        for i in range(self.num_kernels):
            kernel_obj = self.kernels[i]
            sampled_kernels.append(kernel_obj.sample_kernel())
            new_kernels.append(
                self.StochasticKernel(self.kernel_size, self.initial_mu_list[i], self.initial_sigma_list[i])
            )

        self.kernels = new_kernels
        return sampled_kernels

    def get_kernels_params(self):
        return [{'mu': k.threshold, 'sigma': k.sigma} for k in self.kernels]

# ----------------------------
# Grid display of input and feature maps
# ----------------------------
def plot_kernels_grid(image, output_map, pooling_applied=False):
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
        title = f"Kernel {i+1}"
        if pooling_applied:
            title += " (Pooled)"
        axes[i + 1].set_title(title, fontsize=8)
        axes[i + 1].axis('off')

    for j in range(num_maps + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# ----------------------------
# Max pooling using skimage.block_reduce
# ----------------------------
def max_pooling(feature_map, pool_size=(2, 2)):
    return block_reduce(feature_map, block_size=pool_size, func=np.max)

# ----------------------------
# Main execution
# ----------------------------
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data(size=False)
    sample_image = X_train[0].reshape(28, 28)

    kernel_layer = StochasticKernelLayer(num_kernels=9, kernel_size=3)
    sampled_kernels = kernel_layer.sample_kernels()

    convolved_images = []
    pooled_images = []
    pool_size = (2, 2)

    for i, kernel in enumerate(sampled_kernels):
        convolved = convolve2d(sample_image, kernel, mode='same')
        pooled = max_pooling(convolved, pool_size=pool_size)
        convolved_images.append(convolved)
        pooled_images.append(pooled)
        print(f"Kernel {i+1}: conv shape = {convolved.shape}, pooled shape = {pooled.shape}")

    convolved_maps = np.array(convolved_images)
    pooled_maps = np.array(pooled_images)

    print("\n--- Convolution Results ---")
    plot_kernels_grid(sample_image, convolved_maps, pooling_applied=False)

    print("\n--- Max Pooling Results ---")
    plot_kernels_grid(sample_image, pooled_maps, pooling_applied=True)

    print("\nKernel Parameters (μ, σ):")
    for i, params in enumerate(kernel_layer.get_kernels_params()):
        print(f"Kernel {i+1}: mu={params['mu']:.4f}, sigma={params['sigma']:.4f}")
