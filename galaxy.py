import pygame
import numpy as np


class Galaxy:
    def __init__(self, velocity, coordinates, mass, color):
        self.velocity = velocity
        self.coordinates = coordinates
        self.acceleration = [0.0, 0.0]
        self.color = color
        self.mass = mass
        self.constant = -200 * mass

    def print(self, window):
        radius = 1
        pygame.draw.circle(window, self.color, (round(self.coordinates[0]), round(self.coordinates[1])), radius)

    def change_coordinates(self):
        self.coordinates[0] += self.velocity[0]
        self.coordinates[1] += self.velocity[1]

    def motion(self, galaxy_coordinates, galaxy_mass):
        r_3 = ((self.coordinates[0] - galaxy_coordinates[0]) ** 2 + (
                    self.coordinates[1] - galaxy_coordinates[1]) ** 2) ** 1.5
        self.acceleration[0] = self.constant * galaxy_mass * (self.coordinates[0] - galaxy_coordinates[0])/r_3
        self.acceleration[1] = self.constant * galaxy_mass * (self.coordinates[1] - galaxy_coordinates[1])/r_3
        self.velocity[0] += self.acceleration[0]
        self.velocity[1] += self.acceleration[1]



