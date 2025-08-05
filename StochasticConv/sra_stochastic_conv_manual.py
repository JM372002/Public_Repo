import numpy as np
import matplotlib.pyplot as plt
import struct

np.random.seed(1234)

def load_data(size=True):
    def load_images(filename):
        with open(filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            images = image_data.reshape(num_images, rows, cols)
            return images

    def load_labels(filename):
        with open(filename, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    def one_hot_encode_to_array(labels, num_classes=10):
        labels = np.array(labels)
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

    y_train = one_hot_encode_to_array(train_labels, 10)
    y_test = one_hot_encode_to_array(test_labels, 10)

    return X_train, y_train, X_test, y_test

def convolve2d_manual(image, kernel):
    """
    Perform 2D convolution and return the output.

    Parameters:
    - image: (H, W) NumPy array, the input grayscale image.
    - kernel: (m, m) NumPy array, the convolution kernel.

    Returns:
    - result: (H, W) NumPy array, the filtered image.
    """
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2  # Compute padding size

    # Pad the image with zeros (to keep output size same as input)
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Initialize the output image
    result = np.zeros_like(image, dtype=float)

    # Perform convolution
    for i in range(H):
        for j in range(W):
            region = padded_image[i:i+kH, j:j+kW]  # Extract region
            result[i, j] = np.sum(region * kernel)  # Compute dot product

    return result

class StochasticKernel:
    def __init__(self, size=3, T=100):
        """
        Initializes a Stochastic Kernel.

        Args:
            size (int): The size of the square kernel (size x size). Default is 3.
            T (int): The number of Monte Carlo samples for stochastic approximation. Default is 100.
        """
        self.size = size
        self.T = T

        # Initialize threshold, sigma, and kernel with shapes (size, size)
        self.threshold = np.random.uniform(-0.1, 0.1, size=(self.size, self.size))
        self.sigma = np.random.uniform(0.8, 1.2, size=(self.size, self.size))
        self.kernel = np.random.uniform(-1, 1, size=(self.size, self.size))

    def SRA_sigmoid(self, threshold, kernel, sigma):
        """
        Stochastic Resonance Approximation sigmoid using Monte Carlo noise for a single kernel element.

        This function approximates a stochastic sigmoid by averaging over T Monte Carlo samples.
        For each sample, it adds Gaussian noise to the kernel and checks if it exceeds the threshold.
        The output is the mean probability of exceeding the threshold over all T samples.

        Args:
            threshold (float): The threshold value for this kernel element.
            linear_output (float): The current value of this kernel element (before stochastic update).
            sigma (float): The standard deviation (sigma) of the Gaussian noise for this kernel element.

        Returns:
            float: The stochastic sigmoid output for this kernel element, representing the probability
                   of the noisy kernel exceeding the threshold.
        """
        # Generate noise for T Monte Carlo samples. Shape (T,) is sufficient for element-wise operation.
        noise = np.random.normal(loc=0, scale=sigma, size=self.T)

        # Add noise to the kernel element value. Broadcasting makes this element-wise.
        noisy_values = 2 * kernel + noise

        # Calculate the probability of exceeding the threshold
        return 2 * (noisy_values > threshold).mean() - 1

    def stochastic_forward(self):
        """
        Updates the kernel elements stochastically.

        For each element in the kernel, it applies the SRA_sigmoid function to introduce stochastic fluctuations.
        This function should be called each time you want to apply the stochastic kernel, effectively
        randomly fluctuating the kernel elements for each new sample or iteration.

        Returns:
            numpy.ndarray: The updated stochastic kernel.
        """
        new_kernel = np.zeros_like(self.kernel)
        for i in range(self.size):
            for j in range(self.size):
                # Apply stochastic sigmoid to each element of the kernel
                new_kernel[i, j] = self.SRA_sigmoid(
                    self.threshold[i, j], self.kernel[i, j], self.sigma[i, j]
                )
        self.kernel = new_kernel # Update the kernel with the new stochastic values.
        return self.kernel

class StochasticConvLayer:
    def __init__(self, num_kernels, kernel_size=3, T=100):
        self.num_kernels = num_kernels
        self.kernels = [StochasticKernel(size=kernel_size, T=T) for _
                        in range(num_kernels)]

    def forward(self, input_image):
        """
        Performs a forward pass through the Stochastic Convolutional Layer.

        For each kernel in the layer, it:
        1. Stochastically updates the kernel.
        2. Convolves the updated kernel with the input image.

        Args:
            input_image (numpy.ndarray): The input image (or feature map) as a 2D numpy array.
                                         Assumes single channel input for simplicity.

        Returns:
            numpy.ndarray: The output feature maps stacked along the channel dimension (3D array).
                           Shape: (num_kernels, output_height, output_width)
        """
        output_feature_maps = []
        for kernel in self.kernels:
            # 1. Update the kernel stochastically for this forward pass
            # kernel.stochastic_forward()

            # 2. Perform 2D convolution
            feature_map = convolve2d_manual(input_image, kernel.kernel)

            output_feature_maps.append(feature_map)

        # 3. Stack the feature maps along a new dimension (channel dimension)
        return np.stack(output_feature_maps, axis=0)

def plot_kernels():
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
        ax_kernel_output.set_title(f"Kernel {i+1}", fontsize=8) # Smaller fontsize
        ax_kernel_output.axis('off')

    # If the grid is not completely filled, hide any empty subplots
    for i in range(num_output_kernels + 1, grid_rows * grid_cols):
        if grid_rows > 1:
            axes[i // grid_cols, i % grid_cols].axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # Load grayscale image directly
    X_train, y_train, X_test, y_test = load_data(size=False)
    image = X_train[0].reshape(28, 28)

    # Stochastic Kernel
    stochastic_conv_layer = StochasticConvLayer(num_kernels=35,
                                                kernel_size=3, T=100)

    output_map = stochastic_conv_layer.forward(image)

    plot_kernels()