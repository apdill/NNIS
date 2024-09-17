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
        depth,
        mean_soma_radius,
        std_soma_radius,
        D,
        branch_angle,
        mean_branches,
        weave_type=None,
        randomness=0.0,
        curviness=None,
        curviness_magnitude=1.0,
        n_primary_dendrites=4,
        network=None,
        neuron_id=None,
    ):
        self.network = network
        self.position = position
        self.soma = Soma(position, mean_soma_radius, std_soma_radius)
        self.soma_mask = self.soma.create_binary_mask(size=(network.height, network.width))
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
        self.branch_ends = [
            (point, np.arctan2(point[1] - self.position[1], point[0] - self.position[0]))
            for point in start_points
        ]

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
        """
        Add the accepted branches to the dendrite list, update the dendrite mask, and update branch ends.

        Args:
            accepted_branches (list): List of branches to add.
        """
        new_branch_ends = []

        for branch_info in accepted_branches:
            branch_data = branch_info['branch_data']
            points = branch_data['points']
            new_branches = branch_info['new_branches']

            # Update dendrite list
            self.dendrite.dendrite_list.append(branch_data)

            # Update dendrite mask
            coordinates = np.column_stack((points[0], points[1])).astype(np.int32)
            thickness_start = branch_data['thickness_start']
            thickness_end = branch_data['thickness_end']
            thicknesses = np.linspace(thickness_start, thickness_end, len(coordinates))
            thicknesses = np.clip(np.round(thicknesses), 1, None).astype(int)
            for i in range(len(coordinates) - 1):
                cv2.line(
                    self.dendrite_mask,
                    tuple(coordinates[i]),
                    tuple(coordinates[i + 1]),
                    1,
                    thickness=thicknesses[i],
                )

            # Update branch ends with new branches from accepted branches
            new_branch_ends.extend(new_branches)

        # Update self.branch_ends for the next layer
        self.branch_ends = new_branch_ends

    def draw(self, color):
        """
        Draws the neuron by drawing its soma and dendrites.

        Args:
            color: Color used to draw the neuron.
        """
        self.soma.draw(color)
        self.dendrite.draw(color)

    def generate_binary_mask(self):
        """
        Generates a binary mask of the neuron by combining soma and dendrite masks.

        Returns:
            ndarray: Binary mask of the neuron.
        """
        self.neuron_mask = np.logical_or(self.soma_mask, self.dendrite_mask).astype(np.uint8)
        return self.neuron_mask
