import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_channel, features_dim):
        super(Discriminator, self).__init__()
        """
        Discriminator of the GAN.

        Args:
            input_channel: input channel size of the image as scalar
            features_dim: hidden channel size as scalar

        Output:
            out: discriminator outputs as tensor (1x1)
        """

        # TODO: You are asked to implement 5 layer of CNNs for the discrimininator.
        #       You should gradually increase the output channels by multiplying with
        #       two. For example, first cnn should take input_channel as input
        #       and features_dim as output. Second cnn should take features_dim
        #       as input and output features_dim*2, third cnn should output 
        #       features_dim*2, and etc. Final CNN should output channel size of 1.
        #       For conv1, conv2, conv3, conv4 use kernel_size=4, stride=2, padding=1.
        #       For conv5 se kernel_size=4, stride=2, padding=0. Use LeakyRelu with 0.2
        #       as nonlinearity.

        # TODO: implement cnns by gradually increase the output dimension with factor of 2.
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None

        # TODO: last conv output should be 1.
        self.conv5 = None

        # TODO: Implement activations.
        self.leaky_relu = None
        self.sigmoid = None


    def forward(self, x):
        # TODO: Implement your Forward Pass below.

        pass


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        """
        Generator of the GAN.

        Args:
            channels_noise: noise dimension of the input.
            channels_img: output channel as image dimension
            features_g: hidden dimension to be used.

        Output:
            out: generator outputs as tensor (N x channels_img x 64 x 64)
        """

        # TODO: Now you should reconstruct the image by upsampling from a noise vector.
        #       To do that, you should use ConvTranspose2d function of pytorch for 
        #       upsampling. First convT should that channel_noise as input and output
        #       features_g*16. Than you should gradually decrease the output channel size
        #       with factor of two. For example, second convT should output features_g*8,
        #       third should output features_g*4, and etc. Final cnnT should output channels_img.
        #       In first convT, use kernel_size=4, stride=1, padding=0 and use kernel_size=4, 
        #       stride=1, padding=1 for else. You should use relu as nonlinearity.

        # TODO: Impelement 5-layers of ConvT below
        self.conv_transpose1 = None # Input: N x channels_noise x 1 x 1
        self.conv_transpose2 = None
        self.conv_transpose3 = None
        self.conv_transpose4 = None
        self.conv_transpose5 = None # Output: N x channels_img x 64 x 64

        # TODO: Implement activations.
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # TODO: Implement your Forward Pass below.
        pass


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()