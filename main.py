import numpy as np
import pygame, sys, math
from pygame.locals import *
import random
from pygame import gfxdraw
import galaxy
from star import Star


def main():
    window_size = (800, 800)
    white = (255, 255, 255)
    fps = 50
    red = (255, 0, 0)
    black = (0, 0, 0)

    pygame.init()
    pygame.display.set_caption("Galaxies_collision")
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(window_size)


    galaxies = [galaxy.Galaxy([0.2, -0.5], [200, 400], 1, red), galaxy.Galaxy([-0.5, 0.1], [500, 400], 1, red)]
    stars_list = []
    for i in range(3000):
        star = Star(white, 150)
        star.make_circular_galaxy(galaxies[0].coordinates, galaxies[0].velocity, galaxies[0].mass)
        stars_list.append(star)

    for i in range(3000):
        star = Star(white, 150)
        star.make_circular_galaxy(galaxies[1].coordinates, galaxies[1].velocity, galaxies[1].mass)
        stars_list.append(star)

    state = 0
    while True:
        window.fill(black)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                state = 1
                mouse_position2 = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONUP:
                state = 0
        if state == 1:
            difference = [0, 0]
            mouse_position1 = pygame.mouse.get_pos()
            difference[0] = mouse_position2[0] - mouse_position1[0]
            difference[1] = mouse_position2[1] - mouse_position1[1]
            mouse_position2 = pygame.mouse.get_pos()

            for star in stars_list:
                star.x_coordinate -= difference[0]
                star.y_coordinate -= difference[1]
                star.print(window)
            galaxies[0].coordinates[0] -= difference[0]
            galaxies[0].coordinates[1] -= difference[1]
            galaxies[1].coordinates[0] -= difference[0]
            galaxies[1].coordinates[1] -= difference[1]

        else:
            for star in stars_list:
                star.motion(galaxies[0].coordinates, galaxies[0].mass)
                star.motion(galaxies[1].coordinates, galaxies[1].mass)
                star.change_coordinates()
                star.print(window)

            galaxies[0].motion(galaxies[1].coordinates, galaxies[1].mass)
            galaxies[1].motion(galaxies[0].coordinates, galaxies[0].mass)
            galaxies[0].change_coordinates()
            galaxies[1].change_coordinates()
            # galaxies[0].print(window)
            # galaxies[1].print(window)
        pygame.display.update()
        clock.tick(fps)


if __name__ == "__main__":
    main()