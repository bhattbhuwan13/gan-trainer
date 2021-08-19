# Introduction
This python library is a tool to effortlessly train a generator model using pytorch. The usage instructions are provided below.

# Pre-requisites
1. Close this repository using `git clone`
2. Install required libraries from the `requirements.txt` file

# Train a new model
To train a new model use the `train.py` script or the below code(make change wherever required)
```python
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
```

# Load Saved model and make predictions
To load the saved models and generate images use the `predict.py` script or the code below.
Note: The saved models are stored in `./saved_models` directory.

```python

import pathlib

import torch
from gan_trainer.model import Generator

num_generated_images = 4
# save the model
saved_model_location = pathlib.Path.cwd() / 'saved_models'

# make predictions using trained model, we need generator only
generator = Generator(ngpu=0)
generator.load_state_dict(torch.load(saved_model_location / 'generator.pth'))
fixed_noise = torch.randn(num_generated_images, 100, 1, 1)
generated_images = generator(fixed_noise).detach()
print(generated_images.shape)
```

## Contact
If you have any suggestions please feel free to raise an issue or contact bhuwan@bhuwanbhatt.com
