import numpy as np
import random
import math


class Star:
    def __init__(self, color, radius):
        self.color = color
        self.radius = radius
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.x_acceleration = 0.0
        self.y_acceleration = 0.0
        self.x_coordinate = 0.0
        self.y_coordinate = 0.0
        self.constant = 200

    def make_circular_galaxy(self, galaxy_coordinates, galaxy_velocity, galaxy_mass):
        degree = random.uniform(0, 360) * math.pi / 180.0
        insidefactor = float(random.randint(3000, 10000)) / 10000
        insidefactor = insidefactor * insidefactor
        self.x_coordinate = math.sin(degree) * (insidefactor * self.radius) + galaxy_coordinates[0]
        self.y_coordinate = math.cos(degree) * (insidefactor * self.radius) + galaxy_coordinates[1]
        self.x_velocity = galaxy_velocity[0] + (galaxy_coordinates[1] - self.y_coordinate) * ((galaxy_mass * self.constant / (
            ((self.x_coordinate - galaxy_coordinates[0]) ** 2 + (self.y_coordinate - galaxy_coordinates[1]) ** 2) ** 1.5)) ** 0.5)
        self.y_velocity = galaxy_velocity[1] + (self.x_coordinate - galaxy_coordinates[0]) * ((galaxy_mass * self.constant / (
            ((self.x_coordinate - galaxy_coordinates[0]) ** 2 + (self.y_coordinate - galaxy_coordinates[1]) ** 2) ** 1.5)) ** 0.5)

   
