import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# Now, implement NN
from neural_network import NeuralNetwork

nn = NeuralNetwork(784, 128, 10)

print("Training network...")
nn.train(X_train.T, Y_train, epochs=100, learning_rate=0.1)

print("Testing on dev set...")
predictions = nn.predict(X_dev.T)
accuracy = np.mean(predictions == Y_dev)
print(f'Dev Accuracy: {accuracy * 100:.2f}%')

# Load test data
print("Loading test data...")
test_data = pd.read_csv('data/test.csv')
test_data = np.array(test_data) / 255.0

print("Predicting on test set...")
test_predictions = nn.predict(test_data)

# Save submission
submission = pd.DataFrame({'ImageId': np.arange(1, len(test_predictions)+1), 'Label': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")

