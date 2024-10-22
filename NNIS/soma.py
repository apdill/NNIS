# nn_is/soma.py

import numpy as np
import matplotlib.pyplot as plt
import cv2

class Soma:
    """
    Represents the soma (cell body) of a neuron.

    Attributes:
        position (tuple): The (x, y) position of the soma.
        radius (float): The radius of the soma.
        x_soma (ndarray): X-coordinates of the soma boundary.
        y_soma (ndarray): Y-coordinates of the soma boundary.
        fill (bool): Determines whether the soma is filled or outlined.
    """

    def __init__(self, position, mean_radius, std_radius):
        self.position = position
        self.radius = max(np.random.normal(mean_radius, std_radius), 0)
        self.x_soma, self.y_soma = self._generate_soma()
        self.masks = None

    def _generate_soma(self):
        """
        Generates the coordinates of the soma boundary using a parametric equation.

        Returns:
            tuple: Arrays of x and y coordinates.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        sine_variation = np.random.uniform(0, 15) * np.sin(2 * theta)
        gaussian_variation = np.random.normal(0, 2, len(theta))
        ellipse_ratio = np.random.uniform(0.8, 1.2)
        elongation_angle = np.random.uniform(0, 2 * np.pi)

        x_soma = (self.radius + gaussian_variation + sine_variation) * (
            np.cos(theta) * np.cos(elongation_angle)
            - np.sin(theta) * np.sin(elongation_angle) * ellipse_ratio
        ) + self.position[0]
        y_soma = (self.radius + gaussian_variation + sine_variation) * (
            np.sin(theta) * np.cos(elongation_angle)
            + np.cos(theta) * np.sin(elongation_angle) * ellipse_ratio
        ) + self.position[1]

        return x_soma, y_soma

    def draw(self, color, fill):
        """
        Draws the soma using matplotlib based on the neuron's fill state.

        Args:
            color: Color used to draw the soma.
        """
        if fill == True:
            plt.fill(self.x_soma, self.y_soma, color=color)
        else:
            # Ensure the polygon is closed by appending the first point at the end
            plt.plot(
                np.append(self.x_soma, self.x_soma[0]),
                np.append(self.y_soma, self.y_soma[0]),
                color=color,
                linewidth=1
            )

    def create_binary_mask(self, size=(2048, 2048)):
        """
        Creates both filled and outline binary masks of the soma.

        Args:
            size (tuple): The size of the mask.

        Returns:
            dict: A dictionary containing both filled and outline masks.
        """
        mask_filled = np.zeros(size, dtype=np.uint8)
        mask_outline = np.zeros(size, dtype=np.uint8)
        coordinates = np.array([self.x_soma, self.y_soma]).T.astype(np.int32)

        # Create filled mask
        cv2.fillPoly(mask_filled, [coordinates], 1)

        # Create outline mask
        cv2.polylines(mask_outline, [coordinates], isClosed=True, color=1, thickness=1)

        # Store the masks in a dictionary
        masks = {
            'filled': mask_filled,
            'outline': mask_outline
        }
        self.masks = masks

        return masks
