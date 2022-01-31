import pygame
import numpy as np
import random
import math


class Star:
    def __init__(self, color, radius):
        self.color = color
        self.radius = radius
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.coordinates = np.zeros(2)
        self.constant = 100

    def set_parameters(self, galaxy):

        degree = random.uniform(0, 360) * math.pi / 180.0
        insidefactor = float(random.randint(3000, 10000)) ** 2 / 100000000
        self.coordinates[0] = (math.sin(degree) * (insidefactor * self.radius) + galaxy.coordinates[0])
        self.coordinates[1] = (math.cos(degree) * (insidefactor * self.radius) + galaxy.coordinates[1])
        self.velocity = galaxy.velocity + (np.flip(galaxy.coordinates, 0) - np.flip(self.coordinates, 0)) * \
            (galaxy.mass * self.constant / ((self.coordinates - galaxy.coordinates)**2).sum()**1.5)**0.5 * np.array([1, -1])


    def print(self, window):
        window.set_at((np.around(self.coordinates).astype(int)), self.color)
        self.coordinates += self.velocity

    def motion(self, galaxy):
        r_3 = ((self.coordinates - galaxy.coordinates) ** 2).sum()**1.5
        self.acceleration = -self.constant * galaxy.mass * (self.coordinates - galaxy.coordinates) / r_3
        self.velocity += self.acceleration
