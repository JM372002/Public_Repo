import numpy as np
import struct
from tqdm import tqdm


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

def binary_activation(x):
    return np.where(x >= 0, 1.0, -1.0)

def binary_activation_backward(x):
    return np.ones_like(x)

class BinaryNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.1
        self.b2 = np.zeros((output_dim,))

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1.T + self.b1
        self.a1 = binary_activation(self.z1)
        self.z2 = self.a1 @ self.W2.T + self.b2
        return self.z2

    def backward(self, dout, lr):
        dz2 = dout
        dW2 = dz2.T @ self.a1
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2
        dz1 = da1 * binary_activation_backward(self.z1)
        dW1 = dz1.T @ self.x
        db1 = dz1.sum(axis=0)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

def cross_entropy_loss(logits, targets):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    loss = -np.sum(targets * np.log(probs + 1e-8)) / targets.shape[0]
    return loss, probs

def grad_cross_entropy(probs, targets):
    return (probs - targets) / targets.shape[0]

def accuracy(preds, targets):
    return np.mean(preds == np.argmax(targets, axis=1))

def train(model, X_train, y_train, lr, batch_size, epoch):
    total_loss = 0
    indices = np.random.permutation(X_train.shape[0])

    for start in tqdm(range(0, X_train.shape[0], batch_size), desc=f"Epoch {epoch} [Train]", leave=False):
        end = start + batch_size
        x_batch = X_train[indices[start:end]]
        y_batch = y_train[indices[start:end]]

        logits = model.forward(x_batch)
        loss, probs = cross_entropy_loss(logits, y_batch)
        grad = grad_cross_entropy(probs, y_batch)
        model.backward(grad, lr)
        total_loss += loss

    print(f"Epoch {epoch}: Train Loss = {total_loss / (X_train.shape[0] // batch_size):.4f}")

def test(model, X_test, y_test, epoch):
    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)
    acc = accuracy(preds, y_test)
    print(f"Epoch {epoch}: Test Accuracy = {acc:.4f}")
    print("\n")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(size=False)

    model = BinaryNet(input_dim=784, hidden_dim=512, output_dim=10)
    lr = 1e-3
    batch_size = 64
    epochs = 10

    for epoch in range(1, epochs + 1):
        train(model, X_train, y_train, lr, batch_size, epoch)
        test(model, X_test, y_test, epoch)
