import pygame
import numpy as np


class Galaxy:
    def __init__(self, velocity, coordinates, mass, color):
        self._velocity = np.array(velocity, dtype=float)
        self._coordinates = np.array(coordinates, dtype=float)
        self._acceleration = np.zeros((0, 2))
        self._mass = mass
        self._color = color
        self._constants = 1100

    def print(self, window, image):
        self._coordinates += self._velocity
        radius = 7
        #pygame.draw.circle(window, self._color, (round(self._coordinates[0]), round(self._coordinates[1])), radius)
        window.blit(image, self._coordinates)

    def motion(self, galaxy):
        r_3 = ((self._coordinates[0] - galaxy._coordinates[0]) ** 2
               + (self._coordinates[1] - galaxy._coordinates[1]) ** 2) ** 1.5
        self._acceleration = -self._constants * galaxy._mass * (self._coordinates - galaxy._coordinates)/r_3
        self._velocity = self._velocity + self._acceleration



