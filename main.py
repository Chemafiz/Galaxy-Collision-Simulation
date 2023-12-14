import cv2
import numpy as np
import sys, math
import random
from galaxy import Galaxy
from window import Window
from numba import njit, prange
import time
from functools import partial
from random import randint, uniform, random



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



def build_spiral(n_stars, b, r, rot_fac, fuz_fac):
    spiral_stars = []
    fuzz = int(0.030 * abs(r))
    for i in range(n_stars):
        theta = math.radians(-i)
        x = r * math.exp(b * theta) * math.cos(theta - math.pi * rot_fac) - randint(-fuzz, fuzz) * fuz_fac
        y = r * math.exp(b * theta) * math.sin(theta - math.pi * rot_fac) - randint(-fuzz, fuzz) * fuz_fac
        spiral_stars.append([x, y])
    return np.array(spiral_stars)



def split_number(number, parts):
    base_value = number // parts
    remainder = number % parts
    chunks = [base_value for _ in range(parts)]
    for i in range(remainder):
        chunks[i] += 1

    return chunks


def make_spiral_galaxy(stars, galaxies, b, r, fuz_fac):
    params = (
        (r, 2),
        (r, 1.91),
        (-r, 2),
        (-r, -2.09),
        (-r, 0.5),
        (-r, 0.4),
        (-r, -0.5),
        (-r, -0.6),
    )

    n = len(galaxies)
    n_stars = stars.shape[0]

    chunks_galaxies = split_number(n_stars, n)
    chunk_stars = []
    for i in chunks_galaxies:
        chunk_stars.append(split_number(i, 8))

    for i, galaxy in enumerate(galaxies):
        galaxy_stars = []
        for j, param in enumerate(params):
            galaxy_stars.append(build_spiral(chunk_stars[i][j], b=b, r=param[0], rot_fac=param[1], fuz_fac=fuz_fac))
        galaxy_stars = np.concatenate(galaxy_stars)
        stars[i::n, 0] = galaxy_stars[:, 0] + galaxy.coordinates[0]
        stars[i::n, 1] = galaxy_stars[:, 1] + galaxy.coordinates[1]
        stars[i::n, 2] = galaxy.velocity[0] - (galaxy.coordinates[1] - stars[i::n, 1]) * (
                        np.abs(galaxy.mass * G / ((((stars[i::n, :2] - galaxy.coordinates) ** 2).sum(axis=1))** 1.5)) ** 0.5)

        stars[i::n, 3] = galaxy.velocity[1] + (galaxy.coordinates[0] - stars[i::n, 0]) * (
                np.abs(galaxy.mass * G / ((((stars[i::n, :2] - galaxy.coordinates) ** 2).sum(axis=1))** 1.5)) ** 0.5)



def make_spiral_galaxy2(stars, galaxies, R):
    n = len(galaxies)
    n_stars = stars.shape[0] // (n * 4 + 1)
    thickness = 40


    stars[:n_stars, 0] = galaxies[0].coordinates[0] - np.random.randint(1, R, n_stars)
    stars[:n_stars, 1] = np.random.randint(1, thickness, n_stars) + galaxies[0].coordinates[1]

    stars[n_stars:2*n_stars, 0] = galaxies[0].coordinates[0] + np.random.randint(1, R, n_stars)
    stars[n_stars:2*n_stars, 1] = np.random.randint(-thickness, thickness, n_stars) + galaxies[0].coordinates[1]

    stars[2 * n_stars:3 * n_stars, 0] = np.random.randint(-thickness, thickness, n_stars) + galaxies[0].coordinates[0]
    stars[2 * n_stars:3 * n_stars, 1] = galaxies[0].coordinates[1] + np.random.randint(1, R, n_stars)

    stars[3 * n_stars:4 * n_stars, 0] = np.random.randint(-thickness, thickness, n_stars) + galaxies[0].coordinates[0]
    stars[3 * n_stars:4* n_stars, 1] = galaxies[0].coordinates[1] - np.random.randint(1, R, n_stars)

############
    stars[4 * n_stars:5 * n_stars, 0] = galaxies[1].coordinates[0] - np.random.randint(1, R, n_stars)
    stars[4 * n_stars:5 * n_stars, 1] = np.random.randint(1, thickness, n_stars) + galaxies[1].coordinates[1]

    stars[5 * n_stars:6 * n_stars, 0] = galaxies[1].coordinates[0] + np.random.randint(1, R, n_stars)
    stars[5 * n_stars:6 * n_stars, 1] = np.random.randint(-thickness, thickness, n_stars) + galaxies[1].coordinates[1]

    stars[7 * n_stars:8 * n_stars, 0] = np.random.randint(-thickness, thickness, n_stars) + galaxies[1].coordinates[0]
    stars[7 * n_stars:8 * n_stars, 1] = galaxies[1].coordinates[1] + np.random.randint(1, R, n_stars)

    stars[8 * n_stars:, 0] = np.random.randint(-thickness, thickness, n_stars + 1) + galaxies[1].coordinates[0]
    stars[8 * n_stars:, 1] = galaxies[1].coordinates[1] - np.random.randint(1, R, n_stars + 1)



    stars[:4* n_stars, 2] = galaxies[0].velocity[0] + (galaxies[0].coordinates[1] - stars[:4* n_stars, 1]) * (
            np.abs(galaxies[0].mass * G / ((((stars[:4* n_stars, :2] - galaxies[0].coordinates) ** 2).sum(axis=1)) ** 1.5)) ** 0.5)

    stars[:4* n_stars, 3] = galaxies[0].velocity[1] - (galaxies[0].coordinates[0] - stars[:4* n_stars, 0]) * (
            np.abs(galaxies[0].mass * G / ((((stars[:4* n_stars, :2] - galaxies[0].coordinates) ** 2).sum(axis=1)) ** 1.5)) ** 0.5)

    stars[4 * n_stars:, 2] = galaxies[1].velocity[0] + (galaxies[1].coordinates[1] - stars[4 * n_stars:, 1]) * (
            np.abs(galaxies[1].mass * G / (
                        (((stars[4 * n_stars:, :2] - galaxies[1].coordinates) ** 2).sum(axis=1)) ** 1.5)) ** 0.5)

    stars[4 * n_stars:, 3] = galaxies[1].velocity[1] - (galaxies[1].coordinates[0] - stars[4 * n_stars:, 0]) * (
            np.abs(galaxies[1].mass * G / (
                        (((stars[4 * n_stars:, :2] - galaxies[1].coordinates) ** 2).sum(axis=1)) ** 1.5)) ** 0.5)



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


@njit(fastmath=True, parallel=True)
def print_stars(window_matrix, stars, shape):
    for i in prange(stars.shape[0]):
        if 0 <= int(stars[i, 0]) < shape[1] and 0 <= int(stars[i, 1]) < shape[0]:
            window_matrix[int(stars[i, 1]), int(stars[i, 0])] = 255



def move_window(event, x, y, flags, param, stars_params, galaxy_params):
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


def make_galaxy(event, x, y, flags, param, galaxies):
    global speed_x, speed_y, dragging, start_point

    if event == cv2.EVENT_LBUTTONDOWN:
        galaxies.append(Galaxy((0.0, 0.0), (x, y), 3.0, 100))
        start_point = (x, y)
        dragging = True

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        galaxies[-1].velocity = (np.array([x, y]) - galaxies[-1].coordinates) / 100
        speed_x = (x - start_point[0]) / 100
        speed_y = (y - start_point[1]) / 100



current_position = None
dragging = False
speed_x = 0
speed_y = 0
start_point = (0, 0)
G = -100

def main():
    window = Window((720, 1280), "Galaxy Collision")
    cv2.namedWindow(window.title)
    galaxies = []

    callback_with_args = partial(make_galaxy,
                                 galaxies=galaxies)
    cv2.setMouseCallback(window.title, callback_with_args)

    while True:
        for galaxy in galaxies:
            cv2.circle(window.window_matrix, galaxy.coordinates.astype(int), 3, (255, 0, 0), -1)

        if dragging:
            cv2.line(window.window_matrix, (int(start_point[0]), int(start_point[1])),
                     (int(start_point[0] + speed_x * 100), int(start_point[1] + speed_y * 100))
                     ,(255, 255, 255), 1)

        cv2.imshow(window.title, window.window_matrix)
        window.window_matrix.fill(0)

        key = cv2.waitKeyEx(1)
        if key == ord("s"):
            image = cv2.imread("loading.jpg")
            image = cv2.resize(image, (300, 300))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            frame_height, frame_width = window.window_size
            sticker_height, sticker_width = image.shape

            top_left_y = (frame_height - sticker_height) // 2
            top_left_x = (frame_width - sticker_width) // 2

            window.window_matrix[top_left_y:top_left_y + sticker_height, top_left_x:top_left_x + sticker_width] = image
            cv2.imshow(window.title, window.window_matrix)
            cv2.waitKeyEx(1)
            break


    stars_number = 100_000
    stars_params = np.zeros((stars_number, 4))
    galaxy_params = np.zeros((len(galaxies), 4))
    galaxy_const_params = np.zeros((len(galaxies), 2))

    for i, galaxy in enumerate(galaxies):
        galaxy_params[i] = np.concatenate((galaxy.coordinates[:], galaxy.velocity[:]))
        galaxy_const_params[i, :] = galaxy.mass, galaxy.radius


    make_circular_galaxy(stars_params, galaxies, galaxy_const_params[:, 1])
    # make_spiral_galaxy(stars_params, galaxies, 0.6, 200, 1.5)
    # make_spiral_galaxy2(stars_params, galaxies, 200)


    callback_with_args = partial(move_window,
                                 stars_params=stars_params,
                                 galaxy_params=galaxy_params)
    cv2.setMouseCallback(window.title, callback_with_args)

    pause = False
    while True:
        if not pause:
            calculate_stars_coords(stars_params, galaxy_params[:, :2], galaxy_const_params[:, 0])
            calculate_galaxies_coords(galaxy_params, galaxy_const_params[:, 0])

        print_stars(window.window_matrix, stars_params, (720, 1280))
        window.window_matrix = cv2.GaussianBlur(window.window_matrix, (25, 25), 0)

        for i, coords in enumerate(galaxy_params):
            cv2.circle(window.window_matrix, coords[:2].astype(int), 3, (255, 0, 0), -1)

        cv2.imshow(window.title, window.window_matrix)
        window.window_matrix.fill(0)


        key = cv2.waitKeyEx(1)
        if key == ord("q"):
            break
        if key == ord("p"):
            pause = not pause



if __name__ == "__main__":
    main()