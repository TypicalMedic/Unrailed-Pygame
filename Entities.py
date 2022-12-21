import random

import pygame as pg
import Sprites


class train:
    speed = 5
    delay = 10000
    start_delay = delay

    def __init__(self, x, y, col_size, filename):
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

    def move_train(self, time_tick, rail_list, used_rail, station):
        dir_shuffle = [(0, 'up'), (1, 'down'), (2, 'left'), (3, 'right')]
        random.shuffle(dir_shuffle)
        for i in range(len(dir_shuffle)):
            if self.colliders[dir_shuffle[i][0]].collidelist(rail_list) != -1 and \
                    self.tr_dir['col'].collidelist(used_rail) == -1:
                self.tr_dir = self.direction[dir_shuffle[i][1]]
                break
        for i in range(len(dir_shuffle)):
            if self.colliders[dir_shuffle[i][0]].colliderect(station) and \
                    self.tr_dir['col'].collidelist(used_rail) == -1:
                self.tr_dir = self.direction[dir_shuffle[i][1]]
                break

        self.sprite.rect.x += self.tr_dir['dir'][0]
        self.sprite.rect.y += self.tr_dir['dir'][1]
        self.delay += time_tick / (self.speed / 100)

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
