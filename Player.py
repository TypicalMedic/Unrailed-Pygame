import pygame as pg
import Sprites as spr


class Player(spr.sprite):
    """
    class to store player info and change its direction, position
    uses sprite class for aesthetic purposes, might be replaced with rect later
    inherits sprite class
     """
    def __init__(self, col_size, move_del, ally, x=0, y=0, cell_size=20, filename='Assets/player_up.png'):
        """
        Parameters
        ----------
        x : float
            player x spawn position in cells
        y : float
            player y spawn position in cells
        col_size: float
            collider side size in pixels (collider is square-shaped), usually set as grid cell size
        move_del: int
            the amount player moves by one step in pixels, usually set as grid cell size
        filename : str
            file path to the base player sprite (default = Assets/player_up.png)
        ally : Player
            another player agent to communicate with
        cell_size: int
            size of one game cell in pixels
        """
        self.x = x
        self.y = y
        self.comm = [0, 0, 0]
        x = x*cell_size + cell_size/2
        y = y*cell_size + cell_size/2
        spr.sprite.__init__(self, x, y, filename)
        self.col_size = col_size
        self.ally = ally
        self.collider = pg.rect.Rect(x - col_size * 0.5, y - col_size * 1.5, col_size, col_size)
        self.move_delta = move_del
        self.direction = {
            'left': {
                'dir': [-1, 0],
                'pic':  pg.image.load('Assets/player_left.png').convert_alpha()
            },
            'right': {
                'dir': [1, 0],
                'pic':  pg.image.load('Assets/player_right.png').convert_alpha()
            },
            'up': {
                'dir': [0, -1],
                'pic': pg.image.load('Assets/player_up.png').convert_alpha()
            },
            'down': {
                'dir': [0, 1],
                'pic': pg.image.load('Assets/player_down.png').convert_alpha()
            }
        }
        self.pl_dir = self.direction['up']
    """
    function to change collider position and player sprite based on where the player is facing
    """
    def set_dir(self):
        self.image = self.pl_dir['pic']
        self.collider.x = self.rect.x + self.pl_dir['dir'][0] * self.move_delta
        self.collider.y = self.rect.y + self.pl_dir['dir'][1] * self.move_delta

    """
    moves player forward in the direction where it faces
    """
    def set_pos(self):
        self.rect.x += self.pl_dir['dir'][0] * self.move_delta
        self.x += self.pl_dir['dir'][0]
        self.rect.y += self.pl_dir['dir'][1] * self.move_delta
        self.y += self.pl_dir['dir'][1]
