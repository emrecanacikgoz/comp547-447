import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torchvision.utils import save_image


def inference(dataset, model, digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.
    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]

    folder_path = 'generated_imgs'
    if os.path.exists(folder_path) is False:
        os.mkdir(folder_path)
    
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)

        save_image(out, os.path.join(folder_path, f"generated_{digit}_ex{example}.png"))
 

def save_generated_images(folder_path):

    # list all files in the folder and sort them
    files = sorted(os.listdir(folder_path))

    # define the number of examples and number of different numbers
    num_examples = 5  # 0 to 4
    num_numbers = 10  # 0 to 9

    # create a figure with subplots
    fig, axs = plt.subplots(num_numbers, num_examples, figsize=(10, 20))

    # remove axis labels for clarity
    for ax in axs.flat:
        ax.axis('off')

    # load and plot each image in the appropriate subplot
    for file in files:
        if file.endswith(".png"):
            # extracting the number and example index from the file name
            parts = file.split('_')
            number = int(parts[1])  # The number (0-9)
            example = int(parts[-1].split('.')[0].replace("ex", ""))  # The example index (0-4)

            # load image
            img = mpimg.imread(os.path.join(folder_path, file))

            # plot in the right subplot
            axs[number, example].imshow(img)

    plt.tight_layout()
    plt.savefig("generated.png")


