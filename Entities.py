import random

import pygame as pg
import Sprites


class Train:
    speed = 1   # each speed frame train moves 1 pixel
    delay = 50  # ticks to delay train departure
    x = 0   # grid position
    y = 0

    def __init__(self, x, y, col_size, filename, delay_shift=0):
        """
        Class for train behaviour

        :param x: grid x spawn position
        :param y: grid y spawn position
        :param col_size: collider size in pixels
        :param filename: train sprite file path
        :param delay_shift: additional ticks to prolong train delay,
            used when many games launch in one run for correct calculations
        """
        self.delay += delay_shift
        self.start_delay = self.delay   # start delay is needed only before train takes off
        self.sprite = Sprites.sprite(x, y, filename)
        self.col_size = col_size
        self.colliders = [pg.rect.Rect(x - col_size * 0.5, y - col_size * 1.5, col_size, col_size),  # up
                          pg.rect.Rect(x - col_size * 0.5, y + col_size * 0.5, col_size, col_size),  # down
                          pg.rect.Rect(x - col_size * 1.5, y - col_size * 0.5, col_size, col_size),  # left
                          pg.rect.Rect(x + col_size * 0.5, y - col_size * 0.5, col_size, col_size)]  # right
        # dict used to determine which way the train is facing
        self.direction = {
            'left': {
                'dir': [-1, 0],     # vector facing left
                'col': self.colliders[2],   # left collider
                'pic':  pg.image.load('Assets/train_left.png').convert_alpha()  # train left sprite
            },
            'right': {
                'dir': [1, 0],
                'col': self.colliders[3],
                'pic':  pg.image.load('Assets/train_right.png').convert_alpha()
            },
            'up': {
                'dir': [0, -1],
                'col': self.colliders[0],
                'pic':  pg.image.load('Assets/train_up.png').convert_alpha()
            },
            'down': {
                'dir': [0, 1],
                'col': self.colliders[1],
                'pic':  pg.image.load('Assets/train_down.png').convert_alpha()
            }
        }
        self.tr_dir = self.direction['right']   # current train direction

    def move_train(self, path_rail, used_rail, station):
        """
        Moves train forward and changes direction when needed

        :param path_rail: list of rails in path (ordered)
        :param used_rail: list of used rails (ordered)
        :param station: station sprite
        :return: found rail path (bool) and found station (bool)
        """
        # move along the vector
        self.sprite.rect.x += self.tr_dir['dir'][0]
        self.sprite.rect.y += self.tr_dir['dir'][1]
        self.delay += self.speed    # defining when train moves again

        # update colliders
        self.colliders[0].x = self.sprite.rect.x
        self.colliders[0].y = self.sprite.rect.y - self.col_size

        self.colliders[1].x = self.sprite.rect.x
        self.colliders[1].y = self.sprite.rect.y + self.col_size

        self.colliders[2].x = self.sprite.rect.x - self.col_size
        self.colliders[2].y = self.sprite.rect.y

        self.colliders[3].x = self.sprite.rect.x + self.col_size
        self.colliders[3].y = self.sprite.rect.y

        dirs = ['up', 'down', 'left', 'right']
        found_rail_path = False
        found_station = False
        # check every collider for rail path collision. It is found when train reaches the end of the last used rail
        for i in range(len(dirs)):
            if len(path_rail) > 0 and self.colliders[i].colliderect(path_rail[0]) and \
                    not self.tr_dir['col'].colliderect(used_rail[-1]):
                # change direction to where the rail was found
                self.tr_dir = self.direction[dirs[i]]
                self.sprite.image = self.tr_dir['pic']
                found_rail_path = True
                break
        # check every collider for station collision. It is found when train reaches the end of the last used rail
        for i in range(len(dirs)):
            if self.colliders[i].colliderect(station) and \
                    self.tr_dir['col'].collidelist(used_rail) == -1:
                self.tr_dir = self.direction[dirs[i]]
                self.sprite.image = self.tr_dir['pic']
                found_station = True
                break
        return found_rail_path, found_station


class Resource:
    health = 10
    damage_power = 1    # how much damage resource takes when agent faces it each tick
    loot = 1

    def __init__(self, sprite, health, loot):
        """
        Class for resource (tree/ steel) logic

        :param sprite: resource sprite file path
        :param health: how much damage can resource take before dropping loot
        :param loot: how much loot drops after damaging resource
        """
        self.sprite = sprite
        self.health = health
        self.loot = loot

    def damage(self):
        """
        damages the resource and checks whether enough damage was dealt
        :return: true if resource can be collected
        """
        self.health -= self.damage_power
        if self.health <= 0:
            return True
        else:
            return False
