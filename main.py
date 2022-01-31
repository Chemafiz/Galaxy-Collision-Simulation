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
    fps = 30
    red = (255, 0, 0)
    black = (0, 0, 0)

    pygame.init()
    pygame.display.set_caption("Galaxies_collision")
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(window_size)


    galaxies = [galaxy.Galaxy([0.5, -0.5], [200, 400], 1, red), galaxy.Galaxy([-1, 0.1], [500, 400], 1, red)]
    stars_list = []
    for i in range(1500):
        star = Star(white, 100)
        star.set_parameters(galaxies[0])
        stars_list.append(star)

    for i in range(1500):
        star = Star(white, 100)
        star.set_parameters(galaxies[1])
        stars_list.append(star)

    while True:
        window.fill(black)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    sys.exit()

        for star in stars_list:
            star.motion(galaxies[0])
            star.motion(galaxies[1])
            if np.absolute(star.coordinates).sum() > 2000:
                stars_list.remove(star)
            star.print(window)

        galaxies[0].motion(galaxies[1])
        galaxies[1].motion(galaxies[0])
        galaxies[0].print(window)
        galaxies[1].print(window)
        pygame.display.update()
        clock.tick(fps)


if __name__ == "__main__":
    main()