import pathlib

import torch
import torch.nn as nn
import torch.optim as optim

from gan_trainer.model import Discriminator, Generator


# Initialize BCELoss function
criterion = nn.BCELoss()
# Size of z latent vector (i.e. size of generator input)
nz = 100

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_model(num_gpu=0, disc_path=None, gen_path=None):
    """Initializes and returns a discriminator and generator network

    Parameters
    ----------
    num_gpu : int
        The number of gpus available
    disc_path : str
        Path to the discriminator model (optional)
    gen_path : str
        Path to the generator model (optional)

    Returns
    -------
    discriminator :
        The discriminator network
    generator :
        The generator network
    """
    generator = Generator(num_gpu).to(device)
    discriminator = Discriminator(num_gpu).to(device)
    if disc_path:
        discriminator.load_state_dict(torch.load(disc_path))
    if gen_path:
        generator.load_state_dict(torch.load(gen_path))

    return discriminator, generator


def train_model(num_epochs, data_loader, discriminator, generator, learning_rate):
    """Trains the discriminator and generator model

    Parameters
    ----------
    num_epochs : int
        The number of epochs models needs to be trained for
    data_loader :
        data_loader
    discriminator :
        The untrained discriminator model
    generator :
        The untrained generator model
    learning_rate :
        The learning rate for the optimizer

    Returns
    -------
    discriminator :
        The trained discriminator network
    generator :
        The trained generator network
    """

    G_losses = []
    D_losses = []
    iters = 0

    netD, netG = discriminator, generator
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(data_loader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

    return netD, netG  # returns the discriminator and generator network

