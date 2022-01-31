import pygame
import numpy as np


class Galaxy:
    def __init__(self, velocity, coordinates, mass, color):
        self.velocity = np.array(velocity, dtype=float)
        self.coordinates = np.array(coordinates, dtype=float)
        self.acceleration = np.zeros(2)
        self.color = color
        self.mass = mass
        self.constant = -300 * mass

    def print(self, window):
        self.coordinates += self.velocity
        radius = 7
        pygame.draw.circle(window, self.color, (round(self.coordinates[0]), round(self.coordinates[1])), radius)


    def motion(self, galaxy):
        r_3 = ((self.coordinates - galaxy.coordinates) ** 2).sum()**1.5
        self.acceleration = self.constant * galaxy.mass * (self.coordinates - galaxy.coordinates)/r_3
        self.velocity = self.velocity + self.acceleration



