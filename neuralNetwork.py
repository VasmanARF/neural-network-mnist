import numpy as np
import matplotlib.pyplot as plt
from loadNumbers import get_dataSet

# Load the dataset (features and labels)
data, targets = get_dataSet()

# Randomly initialize parameters (weights and biases)
weights_input_hidden = np.random.uniform(-0.5, 0.5, (20, 784))  # Input to hidden layer weights
weights_hidden_output = np.random.uniform(-0.5, 0.5, (10, 20))  # Hidden to output layer weights
bias_hidden = np.zeros((20, 1))  # Bias for hidden layer neurons
bias_output = np.zeros((10, 1))  # Bias for output layer neurons

# Learning rate
learning_rate = 0.01
# Number of epochs for training
total_epochs = 3

# Start training
for epoch in range(total_epochs):
    correct_predictions = 0  # Track correct classifications for this epoch

    # Iterate over each sample and its target
    for sample, target in zip(data, targets):
        # Prepare the input and label
        sample = sample.reshape(784, 1)  # Flatten input sample to a 1D array
        target = target.reshape(10, 1)   # Reshape label to match output layer size

        # Forward pass: Input layer to hidden layer
        hidden_input = bias_hidden + weights_input_hidden @ sample  # Weighted sum at hidden layer
        hidden_output = 1 / (1 + np.exp(-hidden_input))  # Sigmoid activation for hidden layer

        # Forward pass: Hidden layer to output layer
        output_input = bias_output + weights_hidden_output @ hidden_output  # Weighted sum at output layer
        output_result = 1 / (1 + np.exp(-output_input))  # Sigmoid activation for output layer

        correct_predictions += int(np.argmax(output_result) == np.argmax(target))  # Count correct predictions

        # Backpropagation: Calculate gradients for output layer
        output_delta = output_result - target  # Gradient of the cost function w.r.t. output
        weights_hidden_output -= learning_rate * output_delta @ hidden_output.T  # Update weights for hidden-output layer
        bias_output -= learning_rate * output_delta  # Update bias for output layer

        # Backpropagation: Calculate gradients for hidden layer
        hidden_delta = (weights_hidden_output.T @ output_delta) * (hidden_output * (1 - hidden_output))  # Gradient w.r.t hidden layer
        weights_input_hidden -= learning_rate * hidden_delta @ sample.T  # Update weights for input-hidden layer
        bias_hidden -= learning_rate * hidden_delta  # Update bias for hidden layer

    # Output accuracy for the current epoch
    accuracy = (correct_predictions / data.shape[0]) * 100
    print(f"Epoch {epoch + 1}/{total_epochs} - Accuracy: {round(accuracy, 2)}%")

# Interactive prediction on a selected sample
while True:
    idx = int(input("Select an index from dataset to predict: "))
    selected_sample = data[idx]
    plt.imshow(selected_sample.reshape(28, 28), cmap="Blues")

    selected_sample = selected_sample.reshape(784, 1)

    # Forward pass: Input to hidden layer
    hidden_input = bias_hidden + weights_input_hidden @ selected_sample
    hidden_output = 1 / (1 + np.exp(-hidden_input))

    # Forward pass: Hidden to output layer
    output_input = bias_output + weights_hidden_output @ hidden_output
    output_result = 1 / (1 + np.exp(-output_input))

    # Show predicted digit
    plt.title(f"Predicted Digit: {output_result.argmax()}")
    plt.savefig(f"prediction_{idx}.png")  # Save the plot as an image file
    plt.close()  # Close the plot to free up memory
    print(f"Prediction saved as prediction_{idx}.png in directory of script")

    ch = int(input("Enter 1 to choose a number or 0 to close program."))
    if ch == 0:
        break



