# NNIS/examples/basic_usage.py

import numpy as np
import matplotlib.pyplot as plt

from . import Network
from NNIS.utils.file_io import save_masks, save_dataset

def main():
    # Parametersgit 
    network_width = 2048
    network_height = 2048
    num_neurons = 60

    # Neuron-specific parameters with Gaussian distribution
    neuron_params = {
        'depth': 3,
        'mean_soma_radius': 60,
        'std_soma_radius': 15,
        'D': 1.5,
        'branch_angle': np.pi / 4,
        'mean_branches': 1.5,
        'weave_type': 'Gauss',
        'randomness': 0.2,
        'curviness': 'Gauss',
        'curviness_magnitude': 1.5,
        'n_primary_dendrites': 5,
    }

    # Create and generate the network
    network = Network(network_width, network_height, num_neurons, neuron_params, 'nn0')
    network.seed_neurons()
    network.grow_network()

    # Draw the network
    network.draw()

    # Generate and display the binary mask
    network_mask = network.generate_binary_mask()
    plt.figure(figsize=(11, 11))
    plt.imshow(network_mask, cmap='gray')
    plt.axis('off')
    plt.show()

    # Create DataFrame
    nn0_df = network.create_dataset()
    #print(nn0_df)

    # Save masks and DataFrame (uncomment to save)
    #save_masks(network, output_dir='output')
    #save_dataset(network, nn0_df, output_dir='output')

if __name__ == "__main__":
    main()
