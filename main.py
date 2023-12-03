import cv2
import numpy as np
import sys, math
import random
from galaxy import Galaxy
from star import Star
from window import Window
from colors import Colors
from numba import njit, prange
import time
from functools import partial


current_position = None

@njit(fastmath=True)
def calculate_galaxies_coords(galaxy_params, masses):
    a = np.zeros(galaxy_params.shape, dtype=np.float64)
    for i, (coor1, mass1) in enumerate(zip(galaxy_params[:, :2], masses)):
        for j, (coor2, mass2) in enumerate(zip(galaxy_params[i+1:, :2], masses[i+1:])):
            temp = coor1 - coor2
            r = (temp**2).sum()**1.5
            a[i] += G * mass2 * temp / r
            a[j+1+i] += -G * mass1 * temp / r

    galaxy_params[:, 2:] += a
    galaxy_params[:, :2] += galaxy_params[:, 2:]


@njit(fastmath=True)
def calculate_stars_coords(stars, coords, masses):
    a = np.zeros(stars[:, :2].shape, dtype=np.float64)

    for coor, mass in zip(coords, masses):
        r = ((stars[:, :2] - coor)** 2).sum(axis=1) ** 1.5
        a +=  G * mass * (stars[:, :2] - coor) / r[:, np.newaxis]

    stars[:, 2:] += a
    stars[:, :2] += stars[:, 2:]


def make_circular_galaxy(stars, galaxies, R):
    # np.random.seed(42)
    insidefactor = (np.random.randint(2000, 10001, stars.shape[0]) / 10000) ** 3
    random_degrees = np.random.rand(stars.shape[0]) * 2 * np.pi

    n = len(galaxies)
    for i, galaxy in enumerate(galaxies):
        stars[i::n, 0] = np.sin(random_degrees[i::n]) * (insidefactor[i::n] * R[i]) + galaxy.coordinates[0]
        stars[i::n, 1] = np.cos(random_degrees[i::n]) * (insidefactor[i::n] * R[i]) + galaxy.coordinates[1]

        stars[i::n, 2] = galaxy.velocity[0] + (galaxy.coordinates[1] - stars[i::n, 1]) * (
                np.abs(galaxy.mass * G / ((((stars[i::n, :2] - galaxy.coordinates) ** 2).sum(axis=1))** 1.5)) ** 0.5)

        stars[i::n, 3] = galaxy.velocity[1] - (galaxy.coordinates[0] - stars[i::n, 0]) * (
                np.abs(galaxy.mass * G / ((((stars[i::n, :2] - galaxy.coordinates) ** 2).sum(axis=1))** 1.5)) ** 0.5)


@njit(fastmath=True, parallel=False)
def print_stars(window_matrix, stars, shape):
    for i in prange(stars.shape[0]):
        if 0 <= int(stars[i, 0]) < shape[1] and 0 <= int(stars[i, 1]) < shape[0]:
            window_matrix[int(stars[i, 1]), int(stars[i, 0])] = 255

def mouse_callback(event, x, y, flags, param, stars_params, galaxy_params):
    global current_position

    if event == cv2.EVENT_LBUTTONDOWN:
        current_position = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        current_position = None
    elif event == cv2.EVENT_MOUSEMOVE and current_position is not None:
        dx, dy = x - current_position[0], y - current_position[1]
        stars_params[:, :2] += np.array([[dx, dy]])
        galaxy_params[:, :2] += np.array([[dx, dy]])
        current_position = (x, y)


def main():
    window = Window((720, 1280), "Galaxy Collision")

    galaxy1 = Galaxy((0.5, 0.2), (300, 360), 1.0, 200)
    galaxy2 = Galaxy((-0.5, -0.2), (900, 360), 1.0, 100)
    galaxies = [galaxy1, galaxy2]

    stars_number = 100_000
    stars_params = np.zeros((stars_number, 4))
    galaxy_params = np.zeros((len(galaxies), 4))
    galaxy_const_params = np.zeros((len(galaxies), 2))



    for i, galaxy in enumerate(galaxies):
        galaxy_params[i] = np.concatenate((galaxy.coordinates[:], galaxy.velocity[:]))
        galaxy_const_params[i, :] = galaxy.mass, galaxy.radius

    make_circular_galaxy(stars_params, galaxies, galaxy_const_params[:, 1])


    callback_with_args = partial(mouse_callback,
                                 stars_params=stars_params,
                                 galaxy_params=galaxy_params)
    cv2.namedWindow(window.title)
    cv2.setMouseCallback(window.title, callback_with_args)

    temp_total_time = time.time()
    temp_measure_time = [0, 0, 0, 0, 0]
    counter = 0

    while True:
        temp = time.time()
        calculate_stars_coords(stars_params, galaxy_params[:, :2], galaxy_const_params[:, 0])
        temp_measure_time[0] += time.time() - temp

        temp = time.time()
        calculate_galaxies_coords(galaxy_params, galaxy_const_params[:, 0])
        temp_measure_time[1] += time.time() - temp

        temp = time.time()
        print_stars(window.window_matrix, stars_params, (720, 1280))
        temp_measure_time[2] += time.time() - temp

        temp = time.time()
        window.window_matrix = cv2.GaussianBlur(window.window_matrix, (25, 25), 0)
        temp_measure_time[3] += time.time() - temp

        temp = time.time()
        for i, coords in enumerate(galaxy_params):
            cv2.circle(window.window_matrix, coords[:2].astype(int), 3, (255, 0, 0), -1)

        cv2.imshow(window.title, window.window_matrix)
        window.window_matrix.fill(0)
        temp_measure_time[4] += time.time() - temp

        key = cv2.waitKeyEx(1)
        if key == ord("q"):
            break

        counter += 1

    temp_total_time = time.time() - temp_total_time
    temp_measure_time = [t * 100/temp_total_time for t in temp_measure_time]
    print(f"Times = {temp_measure_time}")
    print(f"Process time  = {temp_total_time / counter:.3f}")


if __name__ == "__main__":
    colors = Colors()
    const = 200
    G = -100
    main()
