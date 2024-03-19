import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model_solution import VariationalAutoEncoder
from utils import inference
from utils import save_generated_images
from torchvision import transforms
from torch.utils.data import DataLoader

# Configuration
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM  = 784
H_DIM      = 200
Z_DIM      = 20
NUM_EPOCHS = 5
BATCH_SIZE = 32
LR_RATE    = 3e-4
BETA       = 1

# dataset loading
dataset      = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# model init
model     = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
criterion = nn.BCELoss(reduction="sum")

# train loop
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        # forward pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)

        # compute loss
        reconstruction_loss = criterion(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        loss = reconstruction_loss + BETA*kl_div

        

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())


model = model.to("cpu")
for idx in range(10):
    inference(dataset, model, idx, num_examples=5)

save_generated_images("generated_imgs")