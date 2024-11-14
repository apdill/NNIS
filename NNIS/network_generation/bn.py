import numpy as np
import cv2
import random
from scipy.interpolate import splprep, splev
from skimage.draw import line as draw_line  # For line pixel extraction

def generate_network(network_params, neuron_params, network_id=None):
    network_width = network_params.get('width', 2048)
    network_height = network_params.get('height', 2048)
    num_neurons = network_params.get('num_neurons', 10)
    edge_margin = network_params.get('edge_margin', 100)

    # Create and generate the network
    network = Network(
        width=network_width,
        height=network_height,
        num_neurons=num_neurons,
        neuron_params=neuron_params,
        edge_margin=edge_margin,
        network_id=network_id
    )

    network.seed_neurons()
    network.grow_network()

    return network

class Soma:
    def __init__(self, position, mean_radius, std_radius):
        self.position = position
        self.radius = max(np.random.normal(mean_radius, std_radius), 0)
        self._generate_soma()

    def _generate_soma(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        sine_variation = np.random.uniform(0, 15) * np.sin(2 * theta)
        gaussian_variation = np.random.normal(0, 2, len(theta))
        ellipse_ratio = np.random.uniform(0.8, 1.2)
        elongation_angle = np.random.uniform(0, 2 * np.pi)

        self.x_soma = (self.radius + gaussian_variation + sine_variation) * (
            np.cos(theta) * np.cos(elongation_angle)
            - np.sin(theta) * np.sin(elongation_angle) * ellipse_ratio
        ) + self.position[0]
        self.y_soma = (self.radius + gaussian_variation + sine_variation) * (
            np.sin(theta) * np.cos(elongation_angle)
            + np.cos(theta) * np.sin(elongation_angle) * ellipse_ratio
        ) + self.position[1]

    def create_binary_masks(self, size, neuron_index):
        coordinates = np.array([self.x_soma, self.y_soma]).T.astype(np.int32)

        # Create the filled mask
        mask_filled = np.zeros(size, dtype=np.uint16)
        cv2.fillPoly(mask_filled, [coordinates], neuron_index)

        # Create the outline mask
        mask_outline = np.zeros(size, dtype=np.uint16)
        cv2.polylines(mask_outline, [coordinates], isClosed=True, color=neuron_index, thickness=1)

        return {'filled': mask_filled, 'outline': mask_outline}

class Dendrite:
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
        total_length=500,
        initial_thickness=10
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
        self.total_length = total_length
        self.initial_thickness = initial_thickness
        self.branch_lengths = self._generate_branch_lengths()

    def _generate_branch_lengths(self):
        r = self.mean_branches ** (-1 / self.D)
        branch_lengths = np.zeros(self.depth)
        normalization_factor = self.total_length / sum(r ** i for i in range(self.depth))

        for i in range(self.depth):
            branch_lengths[i] = normalization_factor * r ** i

        return branch_lengths

    def _calculate_thickness(self, distance_from_start, segment_length):
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
        num_points = max(int(self.curviness_magnitude * 10), 2)
        num_control_points = np.random.randint(4, 7)

        t_values = np.linspace(0.2, 0.8, num_control_points - 2)
        t_values = np.concatenate(([0], t_values, [1]))

        dx = x2 - x1
        dy = y2 - y1
        branch_angle = np.arctan2(dy, dx)
        angle = branch_angle + np.pi / 2

        base_x = x1 + t_values * dx
        base_y = y1 + t_values * dy

        radius = length * np.random.uniform(-0.05, 0.05, size=len(t_values))
        perturb_x = base_x + radius * np.cos(angle)
        perturb_y = base_y + radius * np.sin(angle)

        control_x = perturb_x
        control_y = perturb_y

        tck, u = splprep([control_x, control_y], s=0)
        u_fine = np.linspace(0, 1, num_points)
        xs, ys = splev(u_fine, tck)

        perturbation_scale = length / 200
        if self.curviness == 'Gauss':
            xs += np.random.normal(0, perturbation_scale, num_points)
            ys += np.random.normal(0, perturbation_scale, num_points)
        elif self.curviness == 'Uniform':
            xs += np.random.uniform(-perturbation_scale, perturbation_scale, num_points)
            ys += np.random.uniform(-perturbation_scale, perturbation_scale, num_points)

        xs[0], ys[0] = x1, y1
        xs[-1], ys[-1] = x2, y2

        return xs, ys

    def _grow_branch(self, x, y, angle, remaining_depth):
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

class Neuron:
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
        neuron_index=None,
        initial_thickness=10,
        total_length=500,
        pass_through_probability=0.0  # Pass-through probability as a neuron parameter
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
            total_length,
            initial_thickness
        )

        self.neuron_id = neuron_id
        self.neuron_index = neuron_index
        self.current_depth = 0
        self.branch_ends = []
        self.is_growing = True

        # Dictionary to keep track of masks for each depth
        self.depth_masks = {}
        self.pass_through_probability = pass_through_probability  # Store pass-through probability

    def generate_start_points(self):
        x_soma = self.soma.x_soma
        y_soma = self.soma.y_soma

        num_soma_points = len(x_soma)
        base_indices = np.linspace(
            0, num_soma_points - 1, self.dendrite.n_primary_dendrites, endpoint=False
        ).astype(int)

        random_offsets = np.random.randint(
            -num_soma_points // (100 // self.dendrite.n_primary_dendrites // 1.5),
            (100 // self.dendrite.n_primary_dendrites // 1.5) + 1,
            size=self.dendrite.n_primary_dendrites,
        )
        random_indices = (base_indices + random_offsets) % num_soma_points

        start_points = []
        for index in random_indices:
            start_points.append((x_soma[index], y_soma[index]))

        # Initialize branch ends with angles pointing outward from the soma
        for point in start_points:
            dx = point[0] - self.position[0]
            dy = point[1] - self.position[1]
            angle = np.arctan2(dy, dx)
            # Set parent_mask to None since the starting point is the soma
            self.branch_ends.append((point, angle, None))

    def prepare_next_layer(self):
        if self.current_depth >= self.dendrite.depth or not self.branch_ends:
            self.is_growing = False
            return []

        proposed_branches = []

        for start_point, angle, parent_mask in self.branch_ends:
            branch_data, new_branches = self.dendrite._grow_branch(
                start_point[0], start_point[1], angle, self.dendrite.depth - self.current_depth
            )

            if branch_data is not None:
                proposed_branches.append(
                    {
                        'branch_data': branch_data,
                        'start_point': start_point,
                        'parent_mask': parent_mask,
                        'new_branches': new_branches,
                    }
                )

        return proposed_branches

    def add_branches(self, accepted_branches, network_mask_filled, network_mask_outline, thickness_mask):
        new_branch_ends = []
        depth_mask = np.zeros_like(network_mask_filled, dtype=np.uint8)

        for branch_info in accepted_branches:
            branch_data = branch_info['branch_data']
            points = branch_data['points']
            new_branches = branch_info['new_branches']
            parent_mask = branch_info.get('parent_mask')

            thickness_start = int(branch_data['thickness_start'])
            thickness_end = int(branch_data['thickness_end'])
            num_segments = len(points[0]) - 1
            thickness_values = np.linspace(thickness_start, thickness_end, num_segments).astype(int)

            segment_stopped = False
            is_passing_through = False  # Flag to track if branch is passing through

            # Temporary mask for the current branch
            current_branch_mask = np.zeros_like(network_mask_filled, dtype=np.uint8)
            current_branch_thickness = np.zeros_like(thickness_mask)

            for i in range(num_segments):
                x0, y0 = int(points[0][i]), int(points[1][i])
                x1, y1 = int(points[0][i + 1]), int(points[1][i + 1])
                thickness = thickness_values[i]

                # Get line pixels
                rr, cc = draw_line(y0, x0, y1, x1)

                if thickness > 1:
                    # Create a temporary mask for the thick line
                    temp_mask = np.zeros_like(network_mask_filled, dtype=np.uint8)
                    cv2.line(temp_mask, (x0, y0), (x1, y1), color=1, thickness=thickness)
                    ys, xs = np.nonzero(temp_mask)
                else:
                    ys, xs = rr, cc

                # Ensure indices are within bounds
                ys = np.clip(ys, 0, network_mask_filled.shape[0] - 1)
                xs = np.clip(xs, 0, network_mask_filled.shape[1] - 1)

                # Exclude current branch pixels
                mask_current_branch = current_branch_mask[ys, xs] > 0
                mask_own_soma = self.network.somas_mask[ys, xs] == self.neuron_index

                # Exclude parent branch pixels
                if parent_mask is not None:
                    mask_parent_branch = parent_mask[ys, xs] > 0
                else:
                    mask_parent_branch = np.zeros_like(mask_current_branch, dtype=bool)

                # Exclude sibling branches (branches at the same depth)
                sibling_mask = depth_mask[ys, xs] > 0

                # Collision occurs if pixel is occupied, not part of own soma, not part of current branch,
                # not part of parent branch, and not part of sibling branches
                existing_thickness = thickness_mask[ys, xs]
                mask_pixel_occupied = existing_thickness > 0

                mask_collision = (
                    mask_pixel_occupied
                    & (~mask_own_soma)
                    & (~mask_current_branch)
                    & (~mask_parent_branch)
                    & (~sibling_mask)
                )

                collision_occurred = np.any(mask_collision)

                # Mark pixels as part of the current branch and store thickness
                current_branch_mask[ys, xs] = 1
                current_branch_thickness[ys, xs] = thickness

                # Handle collision detection based on is_passing_through flag
                if is_passing_through:
                    if not collision_occurred:
                        # No collision detected, stop passing through
                        is_passing_through = False
                    # Continue growing without collision checks
                else:
                    if collision_occurred:
                        # Decide whether to ignore collision based on pass_through_probability
                        if np.random.rand() < self.pass_through_probability:
                            # Ignore collision and continue growing
                            is_passing_through = True
                        else:
                            # Update the masks up to and including the collision pixel
                            ys_all, xs_all = np.nonzero(current_branch_mask)
                            thickness_mask[ys_all, xs_all] = current_branch_thickness[ys_all, xs_all]
                            network_mask_filled[ys_all, xs_all] = self.neuron_index
                            network_mask_outline[ys_all, xs_all] = self.neuron_index
                            depth_mask[ys_all, xs_all] = 1  # Update the depth mask with the current branch

                            # Branch merges at collision point and stops growing further
                            segment_stopped = True
                            break  # Exit the for loop over segments

            if not segment_stopped:
                # Update the masks for the entire branch
                ys_all, xs_all = np.nonzero(current_branch_mask)
                if ys_all.size > 0:
                    thickness_mask[ys_all, xs_all] = current_branch_thickness[ys_all, xs_all]
                    network_mask_filled[ys_all, xs_all] = self.neuron_index
                    network_mask_outline[ys_all, xs_all] = self.neuron_index
                    depth_mask[ys_all, xs_all] = 1  # Update the depth mask with the current branch

                # Add new branch ends for further growth
                for new_branch in new_branches:
                    end_point, new_angle = new_branch
                    new_branch_ends.append((end_point, new_angle, current_branch_mask))

            # If collision occurred and branch stopped, do not add new branches

        # Store the depth mask for the current depth
        self.depth_masks[self.current_depth] = depth_mask

        self.branch_ends = new_branch_ends

class Network:
    def __init__(self, width, height, num_neurons, neuron_params, edge_margin=100, network_id=None):
        self.width = width
        self.height = height
        self.num_neurons = num_neurons
        self.neuron_params = neuron_params
        self.neurons = []
        self.network_mask = np.zeros((self.height, self.width), dtype=np.uint16)
        self.somas_mask = np.zeros((self.height, self.width), dtype=np.uint16)
        self.network_id = network_id
        self.edge_margin = edge_margin

    def seed_neurons(self):
        # Initialize shared masks
        self.network_mask_filled = np.zeros((self.height, self.width), dtype=np.uint16)
        self.network_mask_outline = np.zeros((self.height, self.width), dtype=np.uint16)
        self.thickness_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.somas_mask = np.zeros((self.height, self.width), dtype=np.uint16)

        for neuron_index in range(self.num_neurons):
            neuron_id = f"{self.network_id}_neuron_{neuron_index + 1}"
            neuron_index_int = neuron_index + 1  # Start from 1

            max_attempts = 1000  # Limit the number of attempts to avoid infinite loops
            attempts = 0
            while attempts < max_attempts:
                # Random position within the network area
                neuron_x = random.uniform(0, self.width)
                neuron_y = random.uniform(0, self.height)
                position = (neuron_x, neuron_y)

                neuron = Neuron(
                    position,
                    **self.neuron_params,
                    network=self,
                    neuron_id=neuron_id,
                    neuron_index=neuron_index_int
                )

                soma_masks = neuron.soma.create_binary_masks(size=(self.height, self.width), neuron_index=neuron_index_int)

                # Check for overlap with existing somas
                overlap = np.any((self.somas_mask > 0) & (soma_masks['filled'] > 0))

                if not overlap:
                    # Update masks with neuron index
                    self.network_mask_filled[soma_masks['filled'] == neuron_index_int] = neuron_index_int
                    self.network_mask_outline[soma_masks['outline'] == neuron_index_int] = neuron_index_int
                    self.thickness_mask[soma_masks['filled'] == neuron_index_int] = neuron.dendrite.initial_thickness
                    self.somas_mask[soma_masks['filled'] == neuron_index_int] = neuron_index_int

                    self.neurons.append(neuron)
                    neuron.generate_start_points()
                    break  # Soma placed successfully, move to next neuron

                attempts += 1

            if attempts == max_attempts:
                print(f"Warning: Could not place neuron {neuron_index + 1} without overlap after {max_attempts} attempts.")

    def grow_network(self):
        growing = True
        while growing:
            growing = False
            for neuron in self.neurons:
                if neuron.is_growing:
                    proposed_branches = neuron.prepare_next_layer()
                    if proposed_branches:
                        neuron.add_branches(
                            proposed_branches,
                            self.network_mask_filled,
                            self.network_mask_outline,
                            self.thickness_mask
                        )
                        growing = True
                    else:
                        neuron.is_growing = False

            # Increment the current depth for all neurons
            for neuron in self.neurons:
                if neuron.is_growing:
                    neuron.current_depth += 1

    def generate_binary_mask(self):
        return {
            'filled': self.network_mask_filled,
            'outline': self.network_mask_outline
        }
