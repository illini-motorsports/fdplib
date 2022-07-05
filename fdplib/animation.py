import pygame
import numpy as np
from fdplib.track_tools import Track
from pygame import gfxdraw

WIDTH = 1650
HEIGHT = 900
FPS = 60
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

    def car(x,y):
        screen.blit(carImg, (x,y))

    def blitRotate(surf, image, pos, originPos, angle):
        # offset from pivot to center
        image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
        offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
        
        # roatated offset from pivot to center
        rotated_offset = offset_center_to_pivot.rotate(-angle)

        # roatetd image center
        rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

        # rotate and blit the image
        surf.blit(rotated_image, rotated_image_rect)
    
        # draw rectangle around the image
        #pygame.draw.rect(surf, (255, 0, 0), (*rotated_image_rect.topleft, *rotated_image.get_size()),2)


    def draw_img(image, x, y, angle):
        rotated_image = pygame.transform.rotate(image, angle) 
        screen.blit(rotated_image, rotated_image.get_rect(center=image.get_rect(topleft=(x, y)).center).topleft)


    coords, yaw = track.coords_from_acc(ret_yaw=True)
    coords[0] += -1*np.min(coords[0]) # | bring all values into positive area
    coords[1] += -1*np.min(coords[1]) # |

    coords[0] *= WIDTH/np.max(coords[0])
    coords[1] *= HEIGHT/np.max(coords[1])

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

        # for x, y in zip(coords[0], coords[1]):
        #     gfxdraw.pixel(screen, int(x), int(y), WHITE)

        if idx+10 >= max_idx:
            idx = 0
        else:
            idx += 10

        pygame.display.flip()       

    pygame.quit()