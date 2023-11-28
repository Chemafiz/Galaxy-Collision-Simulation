import cv2
import numpy as np
import sys, math
import random
from galaxy import Galaxy
from star import Star
from window import Window
from colors import Colors
from numba import njit



@njit(fastmath=True)
def calculate_galaxies_coords(coords, masses):
    a = np.zeros(coords.shape, dtype=np.float64)
    for i, (coor1, mass1) in enumerate(zip(coords, masses)):
        for j, (coor2, mass2) in enumerate(zip(coords[i+1:], masses[i+1:])):
            temp = coor1 - coor2
            r = (temp**2).sum()**1.5
            a[i] += G * mass1 * temp / r
            a[j+1+i] += -G * mass2 * temp / r
    return a


@njit(fastmath=True)
def calculate_stars_coords(stars, coords, masses):
    a = np.zeros(stars[:, :2].shape, dtype=np.float64)

    for coor, mass in zip(coords, masses):
        r = ((stars[:, :2] - coor)** 2).sum(axis=1) ** 1.5
        a +=  G * mass * (stars[:, :2] - coor) / r[:, np.newaxis]
        # print(a)
    #     break



    stars[:, 2:] += a
    stars[:, :2] += stars[:, 2:]






def make_circular_galaxy(stars, galaxies, R):
    # np.random.seed(42)
    insidefactor = (np.random.randint(2000, 10001, stars.shape[0]) / 10000) ** 3
    random_degrees = np.random.rand(stars.shape[0]) * 2 * np.pi
    random_radiuses = np.random.rand(stars.shape[0]) * R
    n = len(galaxies)
    for i, galaxy in enumerate(galaxies):
        stars[i::n, 0] = np.sin(random_degrees[i::n]) * (insidefactor[i::n] * R) + galaxy.coordinates[0]
        stars[i::n, 1] = np.cos(random_degrees[i::n]) * (insidefactor[i::n] * R) + galaxy.coordinates[1]

        stars[i::n, 2] = galaxy.velocity[0] + (galaxy.coordinates[1] - stars[i::n, 1]) * (
                np.abs(galaxy.mass * G / ((((stars[i::n, :2] - galaxy.coordinates) ** 2).sum(axis=1))** 1.5)) ** 0.5)

        stars[i::n, 3] = galaxy.velocity[1] - (galaxy.coordinates[0] - stars[i::n, 0]) * (
                np.abs(galaxy.mass * G / ((((stars[i::n, :2] - galaxy.coordinates) ** 2).sum(axis=1))** 1.5)) ** 0.5)


@njit(fastmath=True)
def print_stars(window_matrix, stars, shape):
    for star in stars:
        if  0 <= int(star[0]) < shape[1] and 0 <= int(star[1]) < shape[0]:
            window_matrix[int(star[1]), int(star[0])] = 255


def main():
    window = Window((720, 1280), "Galaxy Collision")
    galaxy1 = Galaxy((0.0, 0.2), (300, 360), 10.0)
    galaxy2 = Galaxy((0.0, -0.2), (900, 360), 1.0)

    galaxies = [galaxy2]
    galaxies = [galaxy1, galaxy2]

    stars_number = 50_000
    stars = np.zeros((stars_number, 4))


    make_circular_galaxy(stars, galaxies, 200)
    while True:
        galaxies_coords = np.array([galaxy.coordinates for galaxy in galaxies])
        galaxies_masses = np.array([galaxy.mass for galaxy in galaxies])


        calculate_stars_coords(stars, galaxies_coords, galaxies_masses)



        a = calculate_galaxies_coords(galaxies_coords, galaxies_masses)

        print_stars(window.window_matrix, stars, (720, 1280))
        window.window_matrix = cv2.GaussianBlur(window.window_matrix, (25, 25), 0)

        for i, galaxy in enumerate(galaxies):
            galaxy.update_coords(a[i])
            cv2.circle(window.window_matrix, galaxy.coordinates.astype(int), 3, (255, 0, 0), -1)

        cv2.imshow(window.title, window.window_matrix)

        window.window_matrix.fill(0)
        key = cv2.waitKeyEx(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    colors = Colors()
    const = 200
    G = -100
    main()
