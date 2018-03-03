# -*- coding: utf-8 -*-

# Design: 8 discrete actions.
# Constant speed 5.

#TODO Consider More Components.
"""
For moving object
1. Acceleration and inertia in ideal environment (no friction)
2. Acceleration caused by the internal force
3. direction needs to be added for the above two points

For bouding box
1. Adjust the receptive field
2. Continuous Action (including distance and direction)

For colors of background and moving object
1. color bank
"""



# libraries
import pygame
import numpy as np
import math
import time

# import cv2
import os
from scipy.misc import imsave


# if without the available visible devices
#os.environ["SDL_VIDEODRIVER"] = "dummy"

# size of the window
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 640

# size of the ball
TARGET_WIDTH = 32
TARGET_HEIGHT = 32

# no refelection for BBOX.
# Stuck in that direction.
# virtual material having no physical charateristics
BBOX_WIDTH = 96
BBOX_HEIGHT = 96

# TARGET_TIMESTEPS to end
LIMIT_TIMESTEPS = 3000
LIMIT_MISSING_TIMESTEPS = 3

# RGB colors used - black background, white ball and paddles
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# initialize the screen of a given size
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# update the bbox and the object iteratively
def drawRect( center, sizes, color=(0, 0, 255), line_width=0):
    rect_area = pygame.Rect(center[0], center[1], sizes[0], sizes[1])
    pygame.draw.rect(screen, color, rect_area, line_width)
    return rect_area

# for moving box
# return the flag representing touch limit
# if touch limit, choose direction again.
def reflect_limit(t, margin, limit):
    if t  < 0:
        return -t, 1
    if t + margin >= limit:
        return 2*(limit-1)-t-margin, 1
    return t, 0

# for bounding box
def crop_limit(t, margin, limit):
    if t < 0:
        return 0
    if t + margin >= limit:
        return limit-1-margin
    return t

def targetPlace(pos, pos_incre):
    tx, fx = reflect_limit(pos[0]+pos_incre[0], TARGET_WIDTH, WINDOW_WIDTH)
    ty, fy = reflect_limit(pos[1]+pos_incre[1], TARGET_HEIGHT, WINDOW_HEIGHT)
    return (tx, ty), (fx or fy)

def bboxPlace(pos, pos_incre):
    tx = crop_limit(pos[0]+pos_incre[0], BBOX_WIDTH, WINDOW_WIDTH)
    ty = crop_limit(pos[1]+pos_incre[1], BBOX_HEIGHT, WINDOW_HEIGHT)
    return (tx, ty)

# x (-1, 0, 1) y (-1, 0, 1)
# 8 direction and not move
def random_direction():
    # TODO make direction a continuous one
    # TODO distance can vary. Now fix at 6
    # sin(d)*Distance yi cos(d)*Distance xi
    d = np.random.randint(0, 8)*math.radians(45)
    t = np.random.randint(5,10)
    return d, t

def interval_length(t):
    return t[1]-t[0]

def interval_intersection(t1, t2):
    # intersection of two interval
    start = min(t1[0], t2[0])
    end = max(t1[1], t2[1])
    u = (start, end)
    length = (interval_length(t1) + interval_length(t2) - interval_length(u))
    length = max(length, 0)
    return length


def intersection(r1, r2):
    """
    intersection of two rectangles
    :param r1 tuple (left, right, top, bottom), rectangle 1
    :param r2: tuple (left, right, top, bottom), rectangle 2
    :return: the intersection area
    """
    area = interval_intersection(r1[:2], r2[:2]) * interval_intersection(r1[2:], r2[2:])
    return area

def get_rect(lt_pos, sizes):
    rect = (lt_pos[0], lt_pos[0]+sizes[0], lt_pos[1], lt_pos[1]+sizes[1])
    return rect

class MovingBoxTracking:
    def __init__(self, render = False):
        pygame.font.init()
        self.render = render
        self.target_area = float(TARGET_HEIGHT*TARGET_WIDTH)
        self.target_speed = 4
        self.bbox_speed = 9

    # initialize
    def reset(self):
        # top left place
        pygame.event.pump()
        screen.fill(WHITE)

        self.done = False

        self.missing_steps = 0
        self.steps = 0

        self.target_pos = (WINDOW_WIDTH/2., WINDOW_HEIGHT/2.)
        self.target_direction, self.target_times = random_direction()

        self.target_sizes = (TARGET_WIDTH, TARGET_HEIGHT)
        self.target_rect = drawRect(self.target_pos, self.target_sizes)
        self.move_target()

        self.bbox_pos = (WINDOW_WIDTH/2., WINDOW_HEIGHT/2.)
        self.bbox_sizes = (BBOX_WIDTH, BBOX_HEIGHT)
        self.bbox_rect = drawRect(self.bbox_pos, self.bbox_sizes, color=(0, 255, 0), line_width=3)

        rf = self.get_receptive_field()

        if self.render:
            pygame.display.flip()
        return rf

    # update the target position
    def move_target(self):
        if self.target_times <= 0:
            self.target_direction, self.target_times = random_direction()

        pos_incre = (self.target_speed * np.array([math.cos(self.target_direction),
                                                   math.sin(self.target_direction)])).astype("int")
        self.target_pos, flag = targetPlace(self.target_pos, pos_incre)

        if flag:
            self.target_times = 0
        else:
            self.target_times -= 1

    def move_bbox(self, direction):
        pos_incre = (self.bbox_speed * np.array([math.cos(direction),math.sin(direction)])).astype("int")
        self.bbox_pos = bboxPlace(self.bbox_pos, pos_incre)

    # very important part.
    #TODO 1. Automatically choose receptive region ( need to be resize later, scale may need to be considered as one input)
    #TODO 2. Automatically choose bounding box area.
    #https://stackoverflow.com/questions/17267395/how-to-take-screenshot-of-certain-part-of-screen-in-pygame
    # Get the receptive field before
    def get_receptive_field(self):
        rf = pygame.surfarray.array3d(screen.subsurface(self.bbox_rect))[:, :, 0]/255.
        return rf

    # the action is one discrete action 0-7
    def step(self, action):
        if self.done:
            return None, None, True

        pygame.event.pump()
        screen.fill(WHITE)

        direction = action * math.radians(45)
        self.move_bbox(direction)

        intersection_area = intersection(get_rect(self.target_pos, self.target_sizes), get_rect(self.bbox_pos, self.bbox_sizes))
        reward = intersection_area/self.target_area/10.

        if reward == 0:
            self.missing_steps += 1
        self.steps += 1

        # the target moves again
        self.move_target()

        self.target_rect = drawRect(self.target_pos, self.target_sizes)
        self.bbox_rect = drawRect(self.bbox_pos, self.bbox_sizes, color=(0, 255, 0), line_width=3)

        rf = self.get_receptive_field()
        self.measure_end()

        if self.render:
            pygame.display.flip()

        return rf, reward, self.done, None


    def measure_end(self):
        if self.missing_steps > LIMIT_MISSING_TIMESTEPS or self.steps > LIMIT_TIMESTEPS:
            self.done = True
        else:
            self.done = False

#def test():
#    # blue ta
#    env = MovingBoxTracking(render=True)
#    env.reset()
#    while True:
#        action = np.random.randint(0, 8)
#        time.sleep(0.1)
#        obs, rew, done = env.step(action)
#        print(rew)
#        if done:
#            break


#test()

