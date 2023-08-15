import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from Build_batches import split_ecg_to_action_lists


# Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def one_hot_encode(labels, num_classes):
    # Initialize an empty array to store the one-hot encoded labels
    one_hot_labels = np.zeros((len(labels), num_classes))

    # Set the appropriate index to 1 for each label
    for i, label in enumerate(labels):
        one_hot_labels[i, int(label)] = 1

    return one_hot_labels

def train_network(data_array_action):
    print(f"infos of the data: \n {len(data_array_action)}")
    # Prepare the data
    # Extract input data and labels
    input_data = np.array([data for data, _ in data_array_action])
    labels = np.array([label for _, label in data_array_action])

    input_data = input_data.astype(np.float32)  # Convert to float32
    #labels = labels.astype(np.float32)  # Convert to float32

    labels = one_hot_encode(labels, 20)

    # Split the data into training and testing sets
    input_train, input_test, labels_train, labels_test = train_test_split(
        input_data, labels, test_size=0.2, random_state=42)

    # Convert the data to PyTorch tensors
    input_train = torch.from_numpy(input_train)
    input_test = torch.from_numpy(input_test)
    labels_train = torch.from_numpy(labels_train)
    labels_test = torch.from_numpy(labels_test)

    # Set the hyperparameters
    input_size = input_data.shape[1]
    hidden_size = 30
    output_size = 20
    learning_rate = 0.02
    num_epochs = 1000

    # Create an instance of the neural network
    model = NeuralNetwork(input_size, hidden_size, output_size)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(input_train)
        loss = criterion(outputs, torch.argmax(labels_train, dim=1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Test the model
    with torch.no_grad():
        predictions = model(input_test)
        _, predicted_labels = torch.max(predictions, 1)

    # Print the predicted labels
    print("Predicted labels:", predicted_labels)
    print(labels_test)
    print("True labels:", torch.argmax(labels_test, dim=1))

def try_t_sne(data_array_action):
    input_data = np.array([data for data, _ in data_array_action])
    labels = np.array([label for _, label in data_array_action])

    input_data = input_data.astype(np.float32)  # Convert to float32
    labels = labels.astype(np.float32)  # Convert to float32

    # Step 3: Apply t-SNE to reduce high-dimensional data to 2D (or 3D) space
    tsne = TSNE(n_components=2, random_state=42)  # Set n_components to 3 for 3D visualization
    data_tsne = tsne.fit_transform(input_data)

    # Plot the t-SNE visualization
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

if __name__ == "__main__":
    print("Hello")
    # Output file name
    ecg_file = "data/DHM 2/Andrei/transformed_data.csv"
    data_array_action = split_ecg_to_action_lists(ecg_file)
    train_network(data_array_action)
    #try_t_sne(data_array_action)
    print("end")
