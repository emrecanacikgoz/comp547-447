import torch
from torch import nn

# Algo: input img => hidden dim => (mean, std) => parametrization trick => decoder => output-img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        """ Variational AutoEncoder
        Args:
            input_dim (int): input dimension
            h_dim (int): hidden dimension
            z_dim (int): latent dimension
        
        Return: 
            x_reconstructed (torch.Tensor): reconstructed image
            mu (torch.Tensor): mean
            sigma (torch.Tensor): standard deviation
        """
        super().__init__()

        # TODO: Define the Encoder with fc layers
        # image_to_hidden should be a fc layer with input_dim and h_dim
        # hidden_to_mu should be a fc layer with h_dim and z_dim
        # hidden_to_sigma should be a cf layer with h_dim and z_dim
        self.image_to_hidden = None
        self.hidden_to_mu = None
        self.hidden_to_sigma = None

        # TODO: Define the Decoder with fc layers
        # z_to_hidden should be a fc layer with z_dim and h_dim
        # hidden_to_image should be a fc layer with h_dim and reconstruction
        self.z_to_hidden = None
        self.hidden_to_image = None

        # TODO: use relu as activation function
        self.relu = None

    def encode(self, x):
        # TODO: Implement the encoder forward pass
        # x is the input image
        # this function should return mu and sigma
        pass

    def decode(self, z):
        h = self.relu(self.z_to_hidden(z))
        return torch.sigmoid(self.hidden_to_image(h))

    def forward(self, x):
        mu, sigma = self.encode(x)

        # TODO: Implement the reparameterization trick
        # z = mu + sigma*epsilon
        # use torch.randn_like(sigma) to generate epsilon
        epsilon = None
        z_new = None

        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    x = torch.randn(4, 28*28)
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    assert x_reconstructed.shape == x.shape
    print(mu.shape)
    assert mu.shape == (4, 20)
    print(sigma.shape)
    assert sigma.shape == (4, 20)