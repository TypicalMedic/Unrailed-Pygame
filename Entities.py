import random

import pygame as pg
import Sprites


class Train:
    speed = 5
    delay = 100
    x = 0
    y = 0
    def __init__(self, x, y, col_size, filename, delay_shift=0):
        self.delay += delay_shift
        self.start_delay = self.delay
        self.sprite = Sprites.sprite(x, y, filename)
        self.col_size = col_size
        self.colliders = [pg.rect.Rect(x - col_size * 0.5, y - col_size * 1.5, col_size, col_size),  # up
                          pg.rect.Rect(x - col_size * 0.5, y + col_size * 0.5, col_size, col_size),  # down
                          pg.rect.Rect(x - col_size * 1.5, y - col_size * 0.5, col_size, col_size),  # left
                          pg.rect.Rect(x + col_size * 0.5, y - col_size * 0.5, col_size, col_size)]  # right
        self.direction = {
            'left': {
                'dir': [-1, 0],
                'col': self.colliders[2]
            },
            'right': {
                'dir': [1, 0],
                'col': self.colliders[3]
            },
            'up': {
                'dir': [0, -1],
                'col': self.colliders[0]
            },
            'down': {
                'dir': [0, 1],
                'col': self.colliders[1]
            }
        }
        self.tr_dir = self.direction['right']

    def move_train(self, time_tick, path_rail, used_rail, station):
        dirs = ['up', 'down', 'left', 'right']
        for i in range(len(dirs)):
            if len(path_rail) > 0 and self.colliders[i].colliderect(path_rail[0]) and \
                    self.tr_dir['col'].collidelist(used_rail) == -1:
                self.tr_dir = self.direction[dirs[i]]
                break
        for i in range(len(dirs)):
            if self.colliders[i].colliderect(station) and \
                    self.tr_dir['col'].collidelist(used_rail) == -1:
                self.tr_dir = self.direction[dirs[i]]
                break

        self.sprite.rect.x += self.tr_dir['dir'][0]
        self.sprite.rect.y += self.tr_dir['dir'][1]
        self.delay += self.speed

        self.colliders[0].x = self.sprite.rect.x
        self.colliders[0].y = self.sprite.rect.y - self.col_size

        self.colliders[1].x = self.sprite.rect.x
        self.colliders[1].y = self.sprite.rect.y + self.col_size

        self.colliders[2].x = self.sprite.rect.x - self.col_size
        self.colliders[2].y = self.sprite.rect.y

        self.colliders[3].x = self.sprite.rect.x + self.col_size
        self.colliders[3].y = self.sprite.rect.y


class Resource:
    health = 10
    damage_power = 1
    loot = 1

    def __init__(self, sprite, health, loot):
        self.sprite = sprite
        self.health = health
        self.loot = loot

    def damage(self):
        self.health -= self.damage_power
        if self.health <= 0:
            return True
        else:
            return False
