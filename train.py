import pathlib

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from gan_trainer.trainer import initialize_model, train_model


dataroot = '../gan_data'
image_size = 64
batch_size = 256
workers = 4
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

disc, gen = initialize_model(num_gpu=1)

discriminator, generator = train_model(num_epochs=1, data_loader=dataloader,
                                       discriminator=disc, generator=gen,
                                       learning_rate=0.0002)

# save the model
saved_model_location = pathlib.Path.cwd() / 'saved_models'
saved_model_location.mkdir(parents=True, exist_ok=True)
torch.save(generator.state_dict(), saved_model_location / 'generator.pth')
torch.save(discriminator.state_dict(), saved_model_location / 'discriminator.pth')

