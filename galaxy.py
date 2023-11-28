import numpy as np


class Galaxy:
    def __init__(self, velocity, coordinates, mass):
        self.velocity = np.array(velocity, dtype=np.float32)
        self.coordinates = np.array(coordinates, dtype=np.float32)
        self.a = np.array([0.0, 0.0], dtype=np.float32)
        self.mass = mass
        self.G_constant = -200 * mass


    def update_coords(self, a):
        self.velocity += a
        self.coordinates += self.velocity



    def motion(self, galaxy_coordinates, galaxy_mass):
        r_3 = ((self.coordinates[0] - galaxy_coordinates[0]) ** 2 + (
                    self.coordinates[1] - galaxy_coordinates[1]) ** 2) ** 1.5
        self.acceleration[0] = self.constant * galaxy_mass * (self.coordinates[0] - galaxy_coordinates[0])/r_3
        self.acceleration[1] = self.constant * galaxy_mass * (self.coordinates[1] - galaxy_coordinates[1])/r_3
        self.velocity[0] += self.acceleration[0]
        self.velocity[1] += self.acceleration[1]



