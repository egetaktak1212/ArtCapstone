# pygame demo 0 - window only

# 1 - Import packages
import pygame
import pygwidgets
from pygame.locals import *
import sys
from run_tests import run_everything


# 2 - Define constants
connections = [
    [[1, 2], [2, 3]],
    [[4, 5], [5, 6]],
    [[7, 8], [8, 9]],
    [[10, 11], [11, 12]],
    [[14, 15], [15, 16], [14, 16], [22, 14], [22, 16]],
    [[18, 19], [19, 20]]
]

image_path = "ArtCapstone/keypoints/image/cameron.JPG"
original_points_dict, bbox = run_everything(image_path) #get facial points from run_tests.py, which loads model and gets points

image = pygame.image.load(image_path)
width, height = image.get_size()

#scale points based on how we resized image
points_dict = {}
ratio = height/1000
for i in range(1,23):
    points_dict[i] = (int(original_points_dict[i][0])/ratio, int(original_points_dict[i][1])/ratio)


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WINDOW_WIDTH = width
WINDOW_HEIGHT = height
FRAMES_PER_SECOND = 10


i=0

display_height = 1000
display_width = display_height * (width/height)

# 3 - Initialize the world
pygame.init()
window = pygame.display.set_mode((display_width, display_height))
# window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
#
clock = pygame.time.Clock()

# 4 - Load assets: image(s), sounds(s),  etc.
sizedImage = pygame.transform.scale(image, (display_width, display_height))
# 5 - Initialize variables

selected_points = set()
selected_lines = []
dragging_group = False
dragging_point = None
last_mouse_pos = None

drag_points = set()
last_mouse_pos = None

import math

def point_near_line(px, py, x1, y1, x2, y2, threshold=8):
    # distance from point to line segment
    line_len = math.hypot(x2 - x1, y2 - y1)
    if line_len == 0:
        return False

    t = max(0, min(1, ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / line_len**2))
    proj_x = x1 + t*(x2 - x1)
    proj_y = y1 + t*(y2 - y1)

    dist = math.hypot(px - proj_x, py - proj_y)
    return dist < threshold

def get_connected_points(start_points):
    connected = set(start_points)
    changed = True

    while changed:
        changed = False
        for feature in connections:
            for p1, p2 in feature:
                if p1 in connected or p2 in connected:
                    if p1 not in connected or p2 not in connected:
                        connected.add(p1)
                        connected.add(p2)
                        changed = True
    return connected

# 6 - Loop forever
while True:

    # 7 - Check for and handle events
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()

        # mouse pressed
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            for i, point in points_dict.items():
                px, py = point
                if (mx - px)**2 + (my - py)**2 < 8**2:
                    dragging_point = i
                    break

            if dragging_point is None:
                for feature in connections:
                    for p1, p2 in feature:
                        x1, y1 = points_dict[p1]
                        x2, y2 = points_dict[p2]

                        if point_near_line(mx, my, x1, y1, x2, y2):
                            selected_points = get_connected_points({p1, p2})
                            dragging_group = True
                            last_mouse_pos = event.pos
                            break
        # mouse released
        if event.type == pygame.MOUSEBUTTONUP:
            dragging_point = None
            dragging_group = False
            drag_points.clear()

        # mouse drag
        if event.type == pygame.MOUSEMOTION:

            # move single point
            if dragging_point is not None:
                points_dict[dragging_point] = event.pos

            # move group
            if dragging_group:
                mx, my = event.pos
                dx = mx - last_mouse_pos[0]
                dy = my - last_mouse_pos[1]

                for p in selected_points:
                    x, y = points_dict[p]
                    points_dict[p] = (x + dx, y + dy)

                last_mouse_pos = event.pos
                
    # 8 - Do any "per frame" actions

    # 9 - Clear the window
    window.fill(BLACK)

    # 10 - Draw all window elements
    red="#FF0000"
    window.blit(sizedImage, (0,0)) #draw the background first

    for feature in connections:
        for connection in feature:
            start = points_dict[connection[0]]
            end = points_dict[connection[1]]

            pygame.draw.line(window, red, start, end, width=2)
            
    i=0
    for point in points_dict:
        i+=1
        pygame.draw.circle(window, red, points_dict[i], 4)

    # 11 - Update the window
    pygame.display.update()

    # 12 - Slow things down a bit
    clock.tick(FRAMES_PER_SECOND)  # make pygame wait
