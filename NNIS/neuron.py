# nn_is/neuron.py

import numpy as np
import matplotlib.pyplot as plt
import cv2

from .soma import Soma
from .dendrite import Dendrite

class Neuron:
    """
    Represents a neuron, consisting of a soma and dendrites.

    Attributes:
        position (tuple): The (x, y) position of the neuron.
        soma (Soma): The soma of the neuron.
        dendrite (Dendrite): The dendritic tree of the neuron.
        neuron_mask (ndarray): Binary mask of the neuron.
        neuron_id (str): Unique identifier of the neuron.
        current_depth (int): Current depth of dendrite growth.
        branch_ends (list): List of current branch ends for further growth.
        is_growing (bool): Indicates whether the neuron is still growing.
    """

    def __init__(
        self,
        position,
        depth=5,
        mean_soma_radius=10,
        std_soma_radius=2,
        D=2.0,
        branch_angle=np.pi / 4,
        mean_branches=3,
        weave_type='Gauss',
        randomness=0.1,
        curviness='Gauss',
        curviness_magnitude=1.0,
        n_primary_dendrites=4,
        network=None,
        neuron_id=None,
    ):
        self.network = network
        self.position = position
        self.soma = Soma(position, mean_soma_radius, std_soma_radius)
        self.dendrite = Dendrite(
            self.soma,
            depth,
            D,
            branch_angle,
            mean_branches,
            weave_type,
            randomness,
            curviness,
            curviness_magnitude,
            n_primary_dendrites,
        )

        self.dendrite_mask = np.zeros((network.height, network.width), dtype=np.uint8)
        self.neuron_mask = None
        self.neuron_id = neuron_id
        self.current_depth = 0
        self.branch_ends = []
        self.is_growing = True  # Flag to indicate if the neuron is still growing

    def generate_start_points(self):
        """
        Generates the starting points for the primary dendrites and initializes branch ends.
        """
        start_points = self.dendrite._generate_dendrite_start_points()
        # Initialize branch ends with angles pointing outward from the soma
        for point in start_points:
            dx = point[0] - self.position[0]
            dy = point[1] - self.position[1]
            angle = np.arctan2(dy, dx)
            self.branch_ends.append((point, angle))

    def prepare_next_layer(self):
        """
        Prepare the proposed branches for the next layer without updating the dendrite mask.

        Returns:
            list: Proposed branches for the next layer.
        """
        if self.current_depth >= self.dendrite.depth or not self.branch_ends:
            self.is_growing = False
            return []

        proposed_branches = []

        for start_point, angle in self.branch_ends:
            branch_data, new_branches = self.dendrite._grow_branch(
                start_point[0], start_point[1], angle, self.dendrite.depth - self.current_depth
            )

            if branch_data is not None:
                proposed_branches.append(
                    {
                        'branch_data': branch_data,
                        'start_point': start_point,
                        'new_branches': new_branches,  # Include new branch ends
                    }
                )

        return proposed_branches

    def add_branches(self, accepted_branches):
        new_branch_ends = []

        for branch_info in accepted_branches:
            branch_data = branch_info['branch_data']
            points = branch_data['points']
            new_branches = branch_info['new_branches']

            # Update dendrite list
            self.dendrite.dendrite_list.append(branch_data)

            # Draw only the new branch onto the existing mask
            coordinates = np.column_stack((points[0], points[1])).astype(np.int32)
            thicknesses = np.linspace(
                branch_data['thickness_start'],
                branch_data['thickness_end'],
                len(coordinates)
            ).astype(int)

            for i in range(len(coordinates) - 1):
                cv2.line(
                    self.dendrite_mask,
                    tuple(coordinates[i]),
                    tuple(coordinates[i + 1]),
                    1,
                    thickness=int(thicknesses[i])
                )

            # Update branch ends
            new_branch_ends.extend(new_branches)

        # Update self.branch_ends for the next layer
        self.branch_ends = new_branch_ends



    def draw(self, color, mask_type='filled'):
        """
        Draws the neuron using matplotlib.

        Args:
            color: Color used to draw the neuron.
            mask_type (str): 'filled' or 'outline' to specify which mask to use.
        """
        if mask_type == 'filled':
            self.soma.draw(color, fill=True)
            self.dendrite.draw(color, fill=True)
        else:
            self.soma.draw(color, fill=False)
            self.dendrite.draw(color, fill=False)

    def generate_binary_mask(self):
        """
        Generates both filled and outline binary masks of the neuron by combining soma and dendrite masks.

        Returns:
            dict: A dictionary containing both filled and outline neuron masks.
        """
        # Generate soma masks
        soma_masks = self.soma.create_binary_mask(size=(self.network.height, self.network.width))

        # Generate dendrite masks
        dendrite_masks = self.dendrite.create_dendrite_mask(size=(self.network.height, self.network.width))

        # Combine masks for both filled and outline versions
        neuron_mask_filled = np.logical_or(soma_masks['filled'], dendrite_masks['filled']).astype(np.uint8)
        neuron_mask_outline = np.logical_or(soma_masks['outline'], dendrite_masks['outline']).astype(np.uint8)

        # Store the masks
        self.masks = {
            'filled': neuron_mask_filled,
            'outline': neuron_mask_outline
        }

        return self.masks