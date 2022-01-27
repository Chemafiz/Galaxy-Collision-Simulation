import pygame, sys, math
from pygame.locals import *
import random
from pygame import gfxdraw

def main():
    window_size = (1000, 800)
    #white = (255, 255, 255)
    fps = 60
    #red = (255, 0, 0)
    black = (0, 0, 0)
    pygame.init()
    pygame.display.set_caption("Galaxies_collision")
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(window_size)

    while True:
        window.fill(black)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    sys.exit()
        pygame.display.update()
        clock.tick(fps)


if __name__ == "__main__":
    main()