# imports
import torch
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SIZE  = 784 
NUM_CLASSES = 10
EPOCHS      = 3
BATCH_SIZE  = 64
LR          = 1e-3 # 0.001

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(),  download=False)
test_dataset  = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=False)
train_loader  = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

# Model
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        Here we define the layers of the network, we are subclassing and
        inheriting from nn.Module We create two fully connected layers
        Parameters:
            input_size: the size of the input, in this case 784 (28x28)
            num_classes: the number of classes we want to predict, in this case 10 (0-9)
        Returns:
            None
        """
        super(NN, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.relu= nn.ReLU()

    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between.
        Parameters:
            x: mnist images
        Returns:
            out: the output of the network
        """
        
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize network
model = NN(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Train Network
for epoch in range(EPOCHS):

    model.train()
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        x, y = x.to(device=device), y.to(device=device)

        # Forward
        scores = model(x)
        loss = criterion(scores, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    """
    Check accuracy of our trained model given a loader and a model
    Parameters:
        loader: torch.utils.data.DataLoader
            A loader for the dataset you want to check accuracy on
        model: nn.Module
            The model you want to check accuracy on
    Returns:
        acc: float
            The accuracy of the model on the dataset given by the loader
    """

    num_correct = 0
    num_samples = 0
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x, y = x.to(device=device), y.to(device=device)

            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    return num_correct/num_samples

# Check accuracy on training & test to see how good our model
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

# TODO: add training loss/acc plots, play with hyperparameters, try different optimizers, add more layers, etc.