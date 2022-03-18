import numpy as np
import pygame, sys, math
from pygame.locals import *
import random
from pygame import gfxdraw
from galaxy import Galaxy
from star import Star
from menu import Menu
from colors import Colors

colors = Colors()
const = 200


def motion(array, galaxy_coordinates, galaxy_mass):
    r_3 = ((array[:, 0] - galaxy_coordinates[0]) ** 2 + (array[:, 1] - galaxy_coordinates[1]) ** 2) ** 1.5
    array[:, 4] = -const * galaxy_mass * (array[:, 0] - galaxy_coordinates[0]) / r_3
    array[:, 5] = -const * galaxy_mass * (array[:, 1] - galaxy_coordinates[1]) / r_3
    array[:, 2] += array[:, 4]
    array[:, 3] += array[:, 5]


def change_star_coordinates(array):
    array[:, 0] += array[:, 2]
    array[:, 1] += array[:, 3]



def main():
    menu = Menu((800, 800), colors.black, 40, "Galaxies Collision")
    pygame.init()
    pygame.display.set_caption(menu.title)
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(menu.window_size)

    galaxy_1 = Galaxy([0.5, -0.3], [200, 400], 1)
    galaxy_2 = Galaxy([-0.5, 0.3], [600, 400], 1)
    galaxies = [galaxy_1, galaxy_2]

    galaxy_radius = 90
    stars_number = 3500
    stars_array = np.zeros((stars_number*2, 6))
    stars_lists = []

    counter = 0
    for galaxy in galaxies:
        for i in range(stars_number * counter, stars_number * (counter + 1)):
            star = Star(colors.white, galaxy_radius)
            star.make_circular_galaxy(galaxy.coordinates, galaxy.velocity, galaxy.mass)
            stars_array[i][0] = star.x_coordinate
            stars_array[i][1] = star.y_coordinate
            stars_array[i][2] = star.x_velocity
            stars_array[i][3] = star.y_velocity
            stars_array[i][4] = star.x_acceleration
            stars_array[i][5] = star.y_acceleration
            stars_lists.append(star)
        counter = 1

    while True:
        window.fill(colors.black)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    sys.exit()

        for galaxy in galaxies:
            motion(stars_array, galaxy.coordinates, galaxy.mass)
        change_star_coordinates(stars_array)
        for star in range(stars_number * 2):
            window.set_at((round(stars_array[star][0]), round(stars_array[star][1])), colors.white)

        galaxies[0].motion(galaxies[1].coordinates, galaxies[1].mass)
        galaxies[1].motion(galaxies[0].coordinates, galaxies[0].mass)
        galaxies[0].change_coordinates()
        galaxies[1].change_coordinates()
        galaxies[0].print(window)
        galaxies[1].print(window)
        pygame.display.update()
        clock.tick(menu.fps)


if __name__ == "__main__":
    main()