import pygame as pg


class CustomSprite(pg.sprite.Sprite):
    """
    class to simplify sprite render
    """
    def __init__(self, x, y, filename):
        """
        Parameters
        ----------
        x
            x position of the rect in pixels
        y
            y position of the rect in pixels
        filename : str
            file path to the sprite
        """
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load(
            filename).convert_alpha()
        self.rect = self.image.get_rect(center=(x, y))

