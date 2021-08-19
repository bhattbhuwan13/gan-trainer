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
