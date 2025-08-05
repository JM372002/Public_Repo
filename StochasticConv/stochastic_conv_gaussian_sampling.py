import numpy as np
import struct
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.measure import block_reduce # For max pooling

np.random.seed(1234)

def load_data(size=True):
    # Function to load MNIST images
    def load_images(filename):
        with open(filename, 'rb') as f:  # Use regular 'open()' for non-gz files
            # First 16 bytes are metadata
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # Rest is the image data, flat array of bytes
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            # Reshape into array of images (num_images, rows, cols)
            images = image_data.reshape(num_images, rows, cols)
            return images

    # Function to load MNIST labels
    def load_labels(filename):
        with open(filename, 'rb') as f:  # Use regular 'open()' for non-gz files
            # First 8 bytes are metadata
            magic, num_labels = struct.unpack(">II", f.read(8))
            # Rest is the label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    def one_hot_encode_to_array(labels, num_classes=10):
        labels = np.array(labels)
        # Create a 2D array of zeros
        encoded = np.zeros((len(labels), num_classes), dtype=int)
        # Place a 1 in the corresponding position for each label
        encoded[np.arange(len(labels)), labels] = 1
        return encoded

    # Load the training and test sets
    train_images = load_images(r'C:\Users\Josh\Desktop\Datasets\MINST\train-images.idx3-ubyte')
    train_labels = load_labels(r'C:\Users\Josh\Desktop\Datasets\MINST\train-labels.idx1-ubyte')
    test_images = load_images(r'C:\Users\Josh\Desktop\Datasets\MINST\t10k-images.idx3-ubyte')
    test_labels = load_labels(r'C:\Users\Josh\Desktop\Datasets\MINST\t10k-labels.idx1-ubyte')

    # Check shapes of the data
    if size == True:
        print(f"Train Images Shape: {train_images.shape}")
        print(f"Train Labels Shape: {train_labels.shape}")
        print(f"Test Images Shape: {test_images.shape}")
        print(f"Test Labels Shape: {test_labels.shape}")

    # Flatten from (N, 28, 28) to (N, 784) and convert to float
    X_train = train_images.reshape(-1, 784).astype(np.float64)
    X_test = test_images.reshape(-1, 784).astype(np.float64)

    # Normalize to [0,1] range
    X_train /= 255.0
    X_test /= 255.0

    y_train = one_hot_encode_to_array(train_labels, num_classes=10)
    y_test = one_hot_encode_to_array(test_labels, num_classes=10)

    return X_train, y_train, X_test, y_test


class StochasticKernelLayer:
    class StochasticKernel:
        def __init__(self, kernel_size, initial_mu=0.0, initial_sigma=1.0):

            self.kernel_size = kernel_size
            self.threshold = initial_mu
            self.sigma = initial_sigma
            self.weights = np.random.uniform(-1, 1, size=(self.kernel_size, self.kernel_size))

        def sample_kernel(self):
            kernel = np.random.normal(loc=self.threshold, scale=self.sigma,
                                    size=(self.kernel_size, self.kernel_size))
            return kernel

    def __init__(self, num_kernels, kernel_size):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.kernels = []
        self.initial_mu_list = np.random.uniform(-0.1, 0.1, size=num_kernels) # Store initial mu
        self.initial_sigma_list = np.random.uniform(0.8, 1.2, size=num_kernels) # Store initial sigma

        # Create initial StochasticKernel instances
        for i in range(num_kernels):
            self.kernels.append(self.StochasticKernel(kernel_size, self.initial_mu_list[i], self.initial_sigma_list[i]))

    def sample_kernels(self):
        sampled_kernels = []
        new_kernels = [] # List to hold new StochasticKernels

        for i in range(self.num_kernels):
            kernel_obj = self.kernels[i]
            sampled_kernel = kernel_obj.sample_kernel() # Sample from the current kernel
            sampled_kernels.append(sampled_kernel)

            # Create a new StochasticKernel object to replace the old one
            # Re-initialize with the original initial values for each kernel index.
            new_kernel_obj = self.StochasticKernel(self.kernel_size, self.initial_mu_list[i], self.initial_sigma_list[i])
            new_kernels.append(new_kernel_obj)

        self.kernels = new_kernels # Replace the old kernels with the new ones
        return sampled_kernels

    def get_kernels_params(self):
        params = []
        for kernel_obj in self.kernels:
            params.append({'mu': kernel_obj.threshold, 'sigma': kernel_obj.sigma})
        return params

def plot_kernels_grid(image, output_map, pooling_applied=False):
    # Display images: Original image and Output Feature Maps in a grid
    num_output_kernels = output_map.shape[0]

    # Calculate grid dimensions: aim for roughly square grid
    grid_cols = int(np.ceil(np.sqrt(num_output_kernels + 1))) # +1 for original image
    grid_rows = int(np.ceil((num_output_kernels + 1) / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 12)) # Adjust figsize for grid

    # Plot original image in the first subplot
    ax_original = axes[0, 0] if grid_rows > 1 else axes[0] # Handle single row case
    ax_original.imshow(image, cmap='gray')
    ax_original.set_title("Original", fontsize=8) # Smaller fontsize for grid
    ax_original.axis('off')

    # Plot each output feature map in the grid
    for i in range(num_output_kernels):
        row_index = (i + 1) // grid_cols # +1 because original image is in the first slot
        col_index = (i + 1) % grid_cols
        if grid_rows > 1:
            ax_kernel_output = axes[row_index, col_index]
        else:
            ax_kernel_output = axes[col_index] # Handle single row case

        ax_kernel_output.imshow(output_map[i], cmap='gray')
        title = f"Kernel {i+1}"
        if pooling_applied:
            title += " (Pooled)"
        ax_kernel_output.set_title(title, fontsize=8) # Smaller fontsize
        ax_kernel_output.axis('off')

    # If the grid is not completely filled, hide any empty subplots
    for i in range(num_output_kernels + 1, grid_rows * grid_cols):
        if grid_rows > 1:
            axes[i // grid_cols, i % grid_cols].axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def max_pooling(feature_map, pool_size=(2, 2)):
    """
    Applies max pooling to a single feature map.

    Args:
        feature_map (numpy.ndarray): A 2D feature map.
        pool_size (tuple): The size of the pooling window (height, width).

    Returns:
        numpy.ndarray: The max-pooled feature map.
    """
    return block_reduce(feature_map, block_size=pool_size, func=np.max)

if __name__ == '__main__':

    # Load MNIST Data
    X_train, y_train, X_test, y_test = load_data(size=False)

    # Choose the first image from the training set
    sample_image_flat = X_train[0]
    sample_image = sample_image_flat.reshape(28, 28) # Reshape back to 28x28

    # Initialize StochasticKernelLayer
    kernel_layer = StochasticKernelLayer(num_kernels=9, kernel_size=3)

    # Sample kernels from the layer
    sampled_kernels_layer = kernel_layer.sample_kernels()
    convolved_images = []
    pooled_images = [] # List to store max-pooled images
    pool_size = (2, 2) # Define pooling size

    for i, kernel in enumerate(sampled_kernels_layer):
        # Convolution
        convolved_image = convolve2d(sample_image, kernel, mode='same')
        convolved_images.append(convolved_image)

        # Max Pooling
        pooled_image = max_pooling(convolved_image, pool_size=pool_size)
        pooled_images.append(pooled_image)
        print(f"Shape of convolved image {i+1}: {convolved_image.shape}, Shape of pooled image {i+1}: {pooled_image.shape}")


    output_feature_maps = np.array(convolved_images)
    pooled_feature_maps = np.array(pooled_images)


    print("\n--- Convolution Results ---")
    plot_kernels_grid(sample_image, output_feature_maps, pooling_applied=False)

    print("\n--- Max Pooling Results ---")
    plot_kernels_grid(sample_image, pooled_feature_maps, pooling_applied=True)

    kernel_params = kernel_layer.get_kernels_params()
    print("\nKernel Parameters (mu, sigma):")
    for i, params in enumerate(kernel_params):
        print(f"Kernel {i+1}: mu={params['mu']:.4f}, sigma={params['sigma']:.4f}")