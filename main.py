import pygame, sys, math
from pygame.locals import *
import random
from pygame import gfxdraw
import galaxy


def main():
    window_size = (1900, 1000)
    #white = (255, 255, 255)
    fps = 200
    red = (255, 0, 0)
    black = (0, 0, 0)

    pygame.init()
    pygame.display.set_caption("Galaxies_collision")
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(window_size)
    wfiis = pygame.image.load("wfiis.png")
    ja = pygame.image.load("ja.png")
    oceny = pygame.image.load("oceny.png")

    galaxies = [galaxy.Galaxy([0, -1], [600, 400], 1, red), galaxy.Galaxy([0, 1], [1000, 400], 1, red)]
    while True:
        window.fill(black)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    sys.exit()
        window.blit(wfiis, (0, 0))
        galaxies[0].motion(galaxies[1])
        galaxies[1].motion(galaxies[0])
        galaxies[0].print(window, ja)
        galaxies[1].print(window, oceny)
        pygame.display.update()
        clock.tick(fps)


if __name__ == "__main__":
    main()