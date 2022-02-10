# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import time

# Might be machine dependent.
HEAD = '/data/dseita/stylegan1'
GAN = 'karras2019stylegan-ffhq-1024x1024.pkl'


def main():
    start = time.time()

    # Initialize TensorFlow.
    print('Initialize TF...')
    tflib.init_tf()

    # Load pre-trained network. NOTE(daniel): had to change this to pickle load.
    # url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    # with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    print(f'Loading pretrained... (time: {time.time()-start:0.2f}s)')
    net_name = os.path.join(HEAD, GAN)
    with open(net_name, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector. NOTE(daniel): can change integer in RandomState
    batch_size = 10
    rnd = np.random.RandomState(3)
    latents = rnd.randn(batch_size, Gs.input_shape[1])

    # Generate image.
    print(f'Generating image ... (time: {time.time()-start:0.2f}s)')
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save images.
    os.makedirs(config.result_dir, exist_ok=True)
    for i in range(batch_size):
        png_filename = os.path.join(config.result_dir, f'example_{str(i).zfill(2)}.png')
        PIL.Image.fromarray(images[i], 'RGB').save(png_filename)
    print(f'Done! See {config.result_dir} (time: {time.time()-start:0.2f}s)')


if __name__ == "__main__":
    main()
