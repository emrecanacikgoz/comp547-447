"""
Training of DCGAN network on MNIST dataset
REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Discriminator, Generator, initialize_weights
import matplotlib.pyplot as plt

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4 
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),])

# for MNIST channels_img is 1
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# model
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

# TODO: Define optimizer for generator, use Adam.
opt_gen = None

# TODO: Define optimizer for discriminator, use Adam.
opt_disc = None

# TODO: check https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
criterion = nn.BCELoss()

# TODO: Create random noise, use randn with dimension (32, NOISE_DIM, 1, 1)
fixed_noise = None

gen.train()
disc.train()
img_grid_reals, img_grid_fakes = [], []
loss_dics, loss_gens = [], []
for epoch in range(NUM_EPOCHS):

    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        # train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1) # flatten
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)) # Why ones? Check BCELoss doc.
        disc_fake = disc(fake.detach()).reshape(-1) # flattened, detached for generator.
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # Why zeros? Check BCELoss doc.

        # TODO: calculate loss for discriminator
        loss_disc = None

        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # for logging
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
        loss_dics.append(loss_disc)
        loss_gens.append(loss_gen)

        # for visualization
        with torch.no_grad():
            fake = gen(fixed_noise)
            img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
            img_grid_reals.append(img_grid_real)
            img_grid_fakes.append(img_grid_fake)


# plt.figure(figsize=(10,5))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(loss_gens,label="G")
# plt.plot(loss_dics,label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# for i in range(len(img_grid_reals)):
#     plt.imshow(img_grid_fakes)