# nn_is/dendrite.py

import numpy as np
import matplotlib.pyplot as plt
import cv2

class Dendrite:
    """
    Represents the dendritic tree of a neuron.

    Attributes:
        soma (Soma): The soma associated with this dendrite.
        depth (int): The maximum branching depth.
        D (float): Fractal dimension.
        branch_angle (float): Angle between branches.
        mean_branches (float): Mean number of branches.
        weave_type (str): Type of randomization in branch lengths and angles ('Gauss' or 'Uniform').
        randomness (float): Magnitude of randomness in branch lengths and angles.
        curviness (str): Type of intra-branch weaving ('Gauss' or 'Uniform').
        curviness_magnitude (float): Magnitude of intra-branch curviness.
        n_primary_dendrites (int): Number of primary dendrites starting from the soma.
        dendrite_list (list): List of branches.
        total_length (float): Total length of dendrite.
        initial_thickness (float): Initial thickness of dendrite.
        branch_lengths (ndarray): Lengths of branches at each depth.
    """

    def __init__(
        self,
        soma,
        depth,
        D,
        branch_angle,
        mean_branches,
        weave_type=None,
        randomness=0.0,
        curviness=None,
        curviness_magnitude=1.0,
        n_primary_dendrites=4,
    ):
        self.soma = soma
        self.depth = depth
        self.D = D
        self.branch_angle = branch_angle
        self.mean_branches = mean_branches
        self.weave_type = weave_type
        self.randomness = randomness
        self.curviness = curviness
        self.curviness_magnitude = curviness_magnitude
        self.n_primary_dendrites = n_primary_dendrites
        self.dendrite_list = []

        self.total_length = self._scale_total_length()
        self.initial_thickness = self._scale_initial_thickness()
        self.branch_lengths = self._generate_branch_lengths()

    def _scale_total_length(self):
        """
        Scales the total length of the dendrite based on the soma radius.

        Returns:
            float: Total length of the dendrite.
        """
        base_length = 40
        length_variation_factor = 5
        total_length = base_length + (self.soma.radius * length_variation_factor) * np.random.uniform(
            0.8, 1.2
        )
        return max(total_length, 0)

    def _generate_branch_lengths(self):
        """
        Generates branch lengths for each depth based on the fractal dimension.

        Returns:
            ndarray: Array of branch lengths for each depth.
        """
        r = self.mean_branches ** (-1 / self.D)
        branch_lengths = np.zeros(self.depth)
        normalization_factor = self.total_length / sum(r ** i for i in range(self.depth))

        for i in range(self.depth):
            branch_lengths[i] = normalization_factor * r ** i

        return branch_lengths

    def _generate_dendrite_start_points(self):
        """
        Generates starting points on the soma boundary for primary dendrites.

        Returns:
            list: List of (x, y) tuples representing start points.
        """
        start_points = []
        num_soma_points = len(self.soma.x_soma)
        base_indices = np.linspace(0, num_soma_points - 1, self.n_primary_dendrites, endpoint=False).astype(
            int
        )

        random_offsets = np.random.randint(
            -num_soma_points // (100 // self.n_primary_dendrites // 1.5),
            (100 // self.n_primary_dendrites // 1.5) + 1,
            size=self.n_primary_dendrites,
        )
        random_indices = (base_indices + random_offsets) % num_soma_points

        for index in random_indices:
            start_points.append((self.soma.x_soma[index], self.soma.y_soma[index]))

        return start_points

    def _scale_initial_thickness(self):
        """
        Scales the initial thickness of the dendrite based on soma radius and total length.

        Returns:
            float: Initial thickness of the dendrite.
        """
        base_thickness = 1
        thickness_factor = 0.02
        initial_thickness = base_thickness + thickness_factor * (self.soma.radius + self.total_length)
        return max(initial_thickness, 1)

    def _calculate_thickness(self, distance_from_start, segment_length):
        """
        Calculates the thickness at the start and end of a dendrite segment.

        Args:
            distance_from_start (float): Cumulative length from the soma to the start of the segment.
            segment_length (float): Length of the current segment.

        Returns:
            tuple: Thickness at the start and end of the segment.
        """
        proportion_start = 1 - (distance_from_start / self.total_length)
        proportion_end = 1 - ((distance_from_start + segment_length) / self.total_length)

        proportion_start = np.clip(proportion_start, 0, 1)
        proportion_end = np.clip(proportion_end, 0, 1)

        thickness_at_start = self.initial_thickness * (proportion_start) ** (1 / self.D)
        thickness_at_end = self.initial_thickness * (proportion_end) ** (1 / self.D)

        thickness_at_start = max(thickness_at_start, 1)
        thickness_at_end = max(thickness_at_end, 1)

        return thickness_at_start, thickness_at_end

    def intra_branch_weave(self, x1, y1, x2, y2, length):
        """
        Generates intra-branch weaving for curviness.

        Args:
            x1, y1 (float): Start coordinates.
            x2, y2 (float): End coordinates.
            length (float): Length of the branch.

        Returns:
            tuple: Arrays of x and y coordinates with curviness applied.
        """
        num_points = int(self.curviness_magnitude * 10)
        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)

        if self.curviness == 'Gauss':
            perturb_xs = xs + (length // 50) * np.random.normal(0, 1, num_points)
            perturb_ys = ys + (length // 50) * np.random.normal(0, 1, num_points)
        elif self.curviness == 'Uniform':
            perturb_xs = xs + (length // 50) * np.random.uniform(-1, 1, num_points)
            perturb_ys = ys + (length // 50) * np.random.uniform(-1, 1, num_points)
        else:
            perturb_xs = xs
            perturb_ys = ys

        perturb_xs[0], perturb_ys[0] = x1, y1
        perturb_xs[-1], perturb_ys[-1] = x2, y2

        return perturb_xs, perturb_ys

    def _grow_branch(self, x, y, angle, remaining_depth):
        """
        Recursively grows a dendritic branch.

        Args:
            x, y (float): Starting coordinates.
            angle (float): Direction of growth.
            remaining_depth (int): Remaining depth to grow.

        Returns:
            tuple: Branch data and list of new branches to grow.
        """
        if remaining_depth == 0:
            return None, []

        branch_length = self.branch_lengths[self.depth - remaining_depth]
        sum_length = sum(self.branch_lengths[: self.depth - remaining_depth])

        thickness_start, thickness_end = self._calculate_thickness(sum_length, branch_length)

        if self.weave_type == 'Gauss':
            branch_length *= 1 + np.random.normal(0, self.randomness)
            angle += np.random.normal(0, self.randomness)
        elif self.weave_type == 'Uniform':
            branch_length *= 1 + np.random.uniform(-self.randomness, self.randomness)
            angle += np.random.uniform(-self.randomness, self.randomness)

        end_x = x + branch_length * np.cos(angle)
        end_y = y + branch_length * np.sin(angle)

        weave_x, weave_y = self.intra_branch_weave(x, y, end_x, end_y, branch_length)

        branch_data = {
            'points': np.array([weave_x, weave_y]),
            'length': branch_length,
            'depth': self.depth - remaining_depth,
            'thickness_start': thickness_start,
            'thickness_end': thickness_end,
        }

        num_branches = int(np.clip(np.round(np.random.normal(self.mean_branches, 1)), 1, None))
        new_branches = []

        for i in range(num_branches):
            new_angle = angle + self.branch_angle * (i - (num_branches - 1) / 2)
            if self.weave_type == 'Gauss':
                new_angle += np.random.normal(0, self.randomness)
            elif self.weave_type == 'Uniform':
                new_angle += np.random.uniform(-self.randomness, self.randomness)

            new_branches.append(((end_x, end_y), new_angle))

        return branch_data, new_branches

    def draw(self, color):
        """
        Draws the dendritic branches using matplotlib.

        Args:
            color: Color used to draw the dendrites.
        """
        for branch in self.dendrite_list:
            points = branch['points']
            thickness_start = branch['thickness_start']
            thickness_end = branch['thickness_end']
            thicknesses = np.linspace(thickness_start, thickness_end, len(points[0]))

            for i in range(len(points[0]) - 1):
                plt.plot(points[0][i : i + 2], points[1][i : i + 2], color=color, linewidth=thicknesses[i])

    def create_dendrite_mask(self, size=(2048, 2048), fill = True):
        """
        Creates a binary mask of the dendritic branches.

        Args:
            size (tuple): The size of the mask.

        Returns:
            ndarray: A binary mask of the dendrites.
        """
        mask = np.zeros(size, dtype=np.uint8)

        for branch in self.dendrite_list:
            points = branch['points']
            coordinates = np.column_stack((points[0], points[1])).astype(np.int32)
            
            if fill == True:        

                thickness_start = branch['thickness_start']
                thickness_end = branch['thickness_end']
                thicknesses = np.linspace(thickness_start, thickness_end, len(points[0]))
                thicknesses = np.clip(np.round(thicknesses), 1, None).astype(int)
                
                for i in range(len(coordinates) - 1):
                    cv2.line(
                        mask,
                        tuple(coordinates[i]),
                        tuple(coordinates[i + 1]),
                        1,
                        thickness=thicknesses[i],
                    )
            else: 
                for i in range(len(coordinates) - 1):
                    cv2.line(
                        mask,
                        tuple(coordinates[i]),
                        tuple(coordinates[i + 1]),
                        1,
                        thickness=1,  # Set thickness to 1
                    )
        
        return mask
