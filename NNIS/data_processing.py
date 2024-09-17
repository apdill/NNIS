# NNIS/NNIS/data_processing.py

import os
import numpy as np
import matplotlib.pyplot as plt

from .network import Network
from .utils.file_io import save_masks, save_dataframe

def batch_generate_networks(
    num_networks,
    network_width,
    network_height,
    num_neurons_per_network,
    neuron_params,
    output_dir='data',
    network_prefix='nn'
):
    """
    Generates multiple networks, saves images, masks, and DataFrames.

    Args:
        num_networks (int): Number of networks to generate.
        network_width (int): Width of each network.
        network_height (int): Height of each network.
        num_neurons_per_network (int or list): Number of neurons in each network.
        neuron_params (dict): Parameters for neuron creation.
        output_dir (str): Base directory to save outputs.
        network_prefix (str): Prefix for network IDs.
    """
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    dataframes_dir = os.path.join(output_dir, 'dataframes')

    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(dataframes_dir, exist_ok=True)

    # Handle num_neurons_per_network input
    if isinstance(num_neurons_per_network, int):
        num_neurons_list = [num_neurons_per_network] * num_networks
    elif isinstance(num_neurons_per_network, list):
        if len(num_neurons_per_network) != num_networks:
            raise ValueError("Length of num_neurons_per_network list must match num_networks.")
        num_neurons_list = num_neurons_per_network
    else:
        raise TypeError("num_neurons_per_network must be an int or a list of ints.")

    for i in range(num_networks):
        network_id = f"{network_prefix}{i}"
        num_neurons = num_neurons_list[i]

        # Create and generate the network
        network = Network(
            width=network_width,
            height=network_height,
            num_neurons=num_neurons,
            neuron_params=neuron_params,
            network_id=network_id
        )
        network._seed_neurons()
        network.grow_network()

        # Generate and save the binary mask
        network_mask = network.generate_binary_mask()

        # Save network image
        network_image_filename = f"{network_id}_image.tiff"
        network_image_path = os.path.join(images_dir, network_image_filename)
        plt.imsave(network_image_path, network_mask, cmap='gray')

        # Save masks and DataFrame
        save_masks(network, output_dir=masks_dir)
        df = network.create_dataframe()
        save_dataframe(network, df, output_dir=dataframes_dir)

        print(f"Network {network_id} generated and saved.")
