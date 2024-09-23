# NNIS/examples/generate_networks.py

import numpy as np
from NNIS.data_processing import batch_generate_networks

def main():
    # Parameters
    num_networks = 100
    network_width = 2048
    network_height = 2048
    num_neurons_per_network = 10  # Or provide a list for varying neurons per network

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

    # Generate networks
    batch_generate_networks(
        num_networks=num_networks,
        network_width=network_width,
        network_height=network_height,
        num_neurons_per_network=num_neurons_per_network,
        neuron_params=neuron_params,
        output_dir=r"C:\Users\absolute-zero\Desktop\NNIS\examples\example_networks",
        network_prefix='example_net_100neurons_'

    )

if __name__ == "__main__":
    main()
