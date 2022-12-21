import pygame as pg


class sprite(pg.sprite.Sprite):
    def __init__(self, x, y, filename):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load(
            filename).convert_alpha()
        self.rect = self.image.get_rect(center=(x, y))


class player(sprite):

    def __init__(self, x, y, filename, col_size, move_del):
        sprite.__init__(self, x, y, filename)
        self.col_size = col_size
        self.collider = pg.rect.Rect(x - col_size * 0.5, y - col_size * 1.5, col_size, col_size)
        self.move_delta = move_del
        self.direction = {
            'left': {
                'dir': [-1, 0],
                'pic':  pg.image.load('player_left.png').convert_alpha()
            },
            'right': {
                'dir': [1, 0],
                'pic':  pg.image.load('player_right.png').convert_alpha()
            },
            'up': {
                'dir': [0, -1],
                'pic': pg.image.load('player_up.png').convert_alpha()
            },
            'down': {
                'dir': [0, 1],
                'pic': pg.image.load('player_down.png').convert_alpha()
            }
        }
        self.pl_dir = self.direction['up']

    def set_dir(self):
        self.image = self.pl_dir['pic']
        self.collider.x = self.rect.x + self.pl_dir['dir'][0] * self.move_delta
        self.collider.y = self.rect.y + self.pl_dir['dir'][1] * self.move_delta

    def set_pos(self):
        self.rect.x += self.pl_dir['dir'][0] * self.move_delta
        self.rect.y += self.pl_dir['dir'][1] * self.move_delta

