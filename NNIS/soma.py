# your_library_name/soma.py

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
    """

    def __init__(self, position, mean_radius, std_radius):
        self.position = position
        self.radius = max(np.random.normal(mean_radius, std_radius), 0)
        self.x_soma, self.y_soma = self._generate_soma()

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

    def draw(self, color):
        """ 
        Draws the soma using matplotlib.

        Args:
            color: Color used to fill the soma.
        """
        plt.fill(self.x_soma, self.y_soma, color=color)

    def create_binary_mask(self, size=(2048, 2048), fill = True):
        """
        Creates a binary mask of the soma.

        Args:
            size (tuple): The size of the mask.

        Returns:
            ndarray: A binary mask of the soma.
        """
        mask = np.zeros(size, dtype=np.uint8)
        coordinates = np.array([self.x_soma, self.y_soma]).T.astype(np.int32)
        if fill == True:
            cv2.fillPoly(mask, [coordinates], 1)
        else:
             cv2.polylines(mask, [coordinates], isClosed=True, color=1, thickness=1)
        return mask
