# NNIS/NNIS/data_processing.py

import os
import numpy as np
import matplotlib.pyplot as plt

from .network import Network
from .utils.file_io import save_masks, save_dataset


def generate_network(network_id, neuron_params, network_params, output_base_dir = False,):
    """
    Generates a network with the given ID and saves outputs to the specified directory structure.

    Args:
        network_id (str): Unique identifier for the network.
        output_base_dir (str): Base directory for outputs.
        neuron_params (dict): Parameters for neuron creation.
        network_params (dict): Parameters for network creation (e.g., width, height, num_neurons).
    """
    network_width = network_params.get('width', 2048)
    network_height = network_params.get('height', 2048)
    num_neurons = network_params.get('num_neurons', 60)

    # Create and generate the network
    network = Network(network_width, network_height, num_neurons, neuron_params, network_id)
    network.seed_neurons()
    network.grow_network()

    # Generate the binary mask
    network_mask = network.generate_binary_mask()

    network_ds = network.create_dataset()
    
    if output_base_dir:
    # Define output directories
        images_dir = os.path.join(output_base_dir, 'images')
        masks_dir = os.path.join(output_base_dir, 'masks')
        dataframes_dir = os.path.join(output_base_dir, 'dataframes')
    
        save_dataset(network_ds, dataframes_dir=dataframes_dir)
        save_masks(network_ds, images_dir=images_dir, masks_dir=masks_dir)

    return network


def batch_generate_networks(
    num_networks,
    network_width,
    network_height,
    num_neurons_per_network,
    neuron_params,
    output_dir,
    network_prefix
):
    """
    Batch generates networks and saves outputs into the specified directory structure.

    Args:
        num_networks (int): Number of networks to generate.
        network_width (int): Width of each network.
        network_height (int): Height of each network.
        num_neurons_per_network (int or list): Number of neurons per network.
        neuron_params (dict): Parameters for neuron creation.
        output_dir (str): Base directory for outputs.
        network_prefix (str): Prefix for network IDs.
        returns an array of network objects
    """

    networks = []

    # Ensure output directories exist
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    dataframes_dir = os.path.join(output_dir, 'dataframes')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(dataframes_dir, exist_ok=True)

    # If num_neurons_per_network is a single integer, create a list
    if isinstance(num_neurons_per_network, int):
        num_neurons_list = [num_neurons_per_network] * num_networks
    elif isinstance(num_neurons_per_network, list):
        num_neurons_list = num_neurons_per_network
        if len(num_neurons_list) != num_networks:
            raise ValueError("Length of num_neurons_per_network list must match num_networks")
    else:
        raise TypeError("num_neurons_per_network must be an int or a list of ints")

    network_params_base = {
        'width': network_width,
        'height': network_height,
    }

    for i in range(num_networks):
        network_id = f"{network_prefix}{i}"
        print(f"Generating network {network_id}")

        # Update network_params with the number of neurons for this network
        network_params = network_params_base.copy()
        network_params['num_neurons'] = num_neurons_list[i]

        # Generate the network
        networks.append(generate_network(network_id, output_dir, neuron_params, network_params))