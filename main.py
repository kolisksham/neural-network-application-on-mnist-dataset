import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ==========================================
# 1. DATA PREPARATION
# ==========================================
def load_data():
    # Load and Prepare the Data from train.csv
    # Make sure this path matches your Kaggle/local environment
    data = pd.read_csv('data/train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)  # Shuffle before splitting

    # Create Dev (Validation) Set (First 1000 images)
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    # Create Training Set (The rest of the images)
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    return X_train, Y_train, X_dev, Y_dev

# ==========================================
# 2. NEURAL NETWORK CLASS
# ==========================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialized with a slightly larger multiplier (0.1) for better starting weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1 
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    # Using ReLU for the hidden layer for faster, healthier learning
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return z > 0
    
    def softmax(self, z):
        # Subtracting the max for numerical stability (prevents overflow)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1) 
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Create one-hot encoded true labels
        y_onehot = np.zeros((m, self.output_size))
        y_onehot[np.arange(m), y] = 1
        
        # Calculate gradients
        dz2 = output - y_onehot
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def update(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs, learning_rate):
        m = X.shape[0]
        for epoch in range(epochs):
            output = self.forward(X)
            dW1, db1, dW2, db2 = self.backward(X, y, output)
            self.update(dW1, db1, dW2, db2, learning_rate)
            
            # Print progress every 50 epochs
            if epoch % 50 == 0: 
                predictions = np.argmax(output, axis=1)
                accuracy = np.mean(predictions == y) * 100
                
                # Calculate Loss
                y_onehot = np.zeros((m, self.output_size))
                y_onehot[np.arange(m), y] = 1
                loss = -np.mean(np.sum(np.log(output + 1e-8) * y_onehot, axis=1))
                
                print(f'Epoch {epoch:4d} | Loss: {loss:.4f} | Training Accuracy: {accuracy:.2f}%')
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# ==========================================
# 3. VISUALIZATION FUNCTION
# ==========================================
def test_prediction(index, nn, X, Y):
    # Grab the specific image column (784 pixels)
    current_image = X[:, index]
    
    # Predict expects a 2D array of shape (m, 784), so we reshape
    prediction = nn.predict(current_image.reshape(1, -1))[0]
    label = Y[index]
    
    print("-------------------------")
    print(f"Image Index: {index}")
    print(f"Prediction:  {prediction}")
    print(f"True Label:  {label}")
    
    # Reshape the 784 pixels back into a 28x28 grid and scale back to 0-255 for plotting
    image_to_plot = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image_to_plot, interpolation='nearest')
    plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

print("Loading data...")
X_train, Y_train, X_dev, Y_dev = load_data()

# Initialize the network (784 inputs, 128 hidden nodes, 10 outputs)
nn = NeuralNetwork(784, 128, 10)

print("\nTraining network...")
# We pass X_train.T so the shape aligns with (m, 784) inside the class
nn.train(X_train.T, Y_train, epochs=500, learning_rate=0.1)

print("\nTesting on dev set...")
predictions = nn.predict(X_dev.T)
accuracy = np.mean(predictions == Y_dev)
print(f'Dev Accuracy: {accuracy * 100:.2f}%\n')

print("Displaying test predictions...")
# Look at the first 4 images to see how it did
test_prediction(0, nn, X_train, Y_train)
test_prediction(1, nn, X_train, Y_train)
test_prediction(2, nn, X_train, Y_train)
test_prediction(3, nn, X_train, Y_train)