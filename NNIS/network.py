# nn_is/network.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
import xarray as xr
import pandas as pd

from .neuron import Neuron

class Network:
    """
    Represents a network of neurons.

    Attributes:
        width (int): Width of the network.
        height (int): Height of the network.
        num_neurons (int): Number of neurons in the network.
        neuron_params (dict): Parameters for neuron creation.
        neurons (list): List of neurons in the network.
        network_mask (ndarray): Binary mask of the entire network.
        somas_mask (ndarray): Binary mask of all somas.
        network_dendrites_mask (ndarray): Binary mask of all dendrites.
        network_id (str): Unique identifier for the network.
    """

    def __init__(self, width, height, num_neurons, neuron_params, network_id):
        self.width = width
        self.height = height
        self.num_neurons = num_neurons
        self.neuron_params = neuron_params
        self.neurons = []
        self.network_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.somas_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.network_dendrites_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.network_id = network_id

    def seed_neurons(self):
        """
        Seeds neurons in the network, ensuring that no two somas overlap.
        """
        for neuron_index in range(self.num_neurons):
            max_attempts = 100  # Limit the number of attempts to avoid infinite loops
            attempts = 0

            while attempts < max_attempts:
                # Generate a random position for the neuron
                position = (np.random.uniform(0, self.width), np.random.uniform(0, self.height))
                neuron_id = f"{self.network_id}_neuron_{neuron_index + 1}"

                # Create a new neuron object with the fill attribute
                neuron = Neuron(
                    position,
                    **self.neuron_params,
                    network=self,
                    neuron_id=neuron_id
                )

                # Create a binary mask of the soma
                new_soma_mask = neuron.soma.create_binary_mask(size=(self.height, self.width))

                # Check if there is an overlap with any existing somas
                overlap = np.any(np.logical_and(self.somas_mask, new_soma_mask))

                if not overlap:
                    # If there is no overlap, add the neuron to the network
                    self.neurons.append(neuron)
                    self.somas_mask = np.logical_or(self.somas_mask, new_soma_mask).astype(np.uint8)
                    neuron.generate_start_points()
                    break  # Exit the while loop and move to the next neuron

                attempts += 1

            if attempts == max_attempts:
                print(
                    f"Warning: Could not place neuron {neuron_index + 1} without overlap after {max_attempts} attempts."
                )

    def grow_network(self):
        """
        Grows the dendrites of all neurons in the network layer by layer.
        """
        growing = True
        while growing:
            growing = False
            # Collect proposed branches from all neurons
            for neuron in self.neurons:
                if neuron.is_growing:
                    proposed_branches = neuron.prepare_next_layer()
                    if proposed_branches:
                        # Add branches using the neuron's internal fill state
                        neuron.add_branches(proposed_branches)
                        growing = True
                    else:
                        neuron.is_growing = False

            # Update the network dendrite mask with all branches
            self.network_dendrites_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            for neuron in self.neurons:
                self.network_dendrites_mask = np.logical_or(
                    self.network_dendrites_mask, neuron.dendrite_mask
                ).astype(np.uint8)

            # Increment the current depth for all neurons
            for neuron in self.neurons:
                if neuron.is_growing:
                    neuron.current_depth += 1

    def generate_binary_mask(self):
        """
        Generates both filled and outline binary masks of the entire network by combining neuron masks.

        Returns:
            dict: A dictionary containing both filled and outline network masks.
        """
        network_mask_filled = np.zeros((self.height, self.width), dtype=np.uint8)
        network_mask_outline = np.zeros((self.height, self.width), dtype=np.uint8)

        for neuron in self.neurons:
            neuron_masks = neuron.generate_binary_mask()

            network_mask_filled = np.logical_or(network_mask_filled, neuron_masks['filled']).astype(np.uint8)
            network_mask_outline = np.logical_or(network_mask_outline, neuron_masks['outline']).astype(np.uint8)

        # Store the masks in a dictionary
        self.masks = {
            'filled': network_mask_filled,
            'outline': network_mask_outline
        }

        return self.masks

    def draw(self, mask_type='filled'):
        """
        Draws the network of neurons using matplotlib.

        Args:
            mask_type (str): 'filled' or 'outline' to specify which mask to use.
        """
        plt.figure(figsize=(12, 12))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

        for neuron in self.neurons:
            color = np.random.rand(3)
            neuron.draw(color=color, mask_type=mask_type)

        plt.axis('equal')
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.show()

    def create_dataset(self):
        """
        Creates an xarray Dataset containing both filled and outline masks for the network and individual neurons,
        and attaches network parameters as attributes.

        Returns:
            xr.Dataset: Dataset with network data and parameters.
        """
        # Define the coordinates (dimensions)
        coords = {'y': np.arange(self.height), 'x': np.arange(self.width)}

        # Initialize the data variables dictionary
        data_vars = {}

        # Generate masks
        network_masks = self.generate_binary_mask()

        # Add the network masks
        data_vars[f'{self.network_id}_network_mask_filled'] = (('y', 'x'), network_masks['filled'])
        data_vars[f'{self.network_id}_network_mask_outline'] = (('y', 'x'), network_masks['outline'])

        # Add neuron masks
        for neuron in self.neurons:
            neuron_masks = neuron.masks
            data_vars[f'{neuron.neuron_id}_filled'] = (('y', 'x'), neuron_masks['filled'])
            data_vars[f'{neuron.neuron_id}_outline'] = (('y', 'x'), neuron_masks['outline'])

        # Create the xarray Dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Attach network parameters as attributes
        # Flatten the network_params dictionary
        for key, value in self.neuron_params.items():
            ds.attrs[key] = value

        # Attach other network attributes
        ds.attrs['network_width'] = self.width
        ds.attrs['network_height'] = self.height
        ds.attrs['num_neurons'] = self.num_neurons
        ds.attrs['network_id'] = self.network_id

        return ds
