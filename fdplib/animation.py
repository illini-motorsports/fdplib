import pygame
import numpy as np
from fdplib.track_tools import Track

WIDTH = 1680
HEIGHT = 1000
FPS = 60
STEP = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def sim_track(track: Track):
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("<Your game>")
    clock = pygame.time.Clock()

    carImg = pygame.image.load("/Users/collin/code/fdplib/assets/car_sprite.png")

    def draw_img(image, x, y, angle):
        rotated_image = pygame.transform.rotate(image, angle) 
        screen.blit(rotated_image, rotated_image.get_rect(center=image.get_rect(topleft=(x, y)).center).topleft)


    coords, yaw = track.coords_from_acc(ret_yaw=True)
    coords[0] += -1*np.min(coords[0]) # | bring all values into positive area
    coords[1] += -1*np.min(coords[1]) # |

    coords[0] *= (WIDTH * 0.9)/np.max(coords[0])
    coords[1] *= (HEIGHT * 0.9)/np.max(coords[1])

    idx = 0
    max_idx = len(coords[0])
    
    running = True
    while running:

        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        screen.fill(BLACK)

        draw_img(carImg, int(coords[0][idx]), int(coords[1][idx]), yaw[idx])

        if idx+STEP >= max_idx:
            idx = 0
        else:
            idx += STEP

        pygame.display.flip()       

    pygame.quit()