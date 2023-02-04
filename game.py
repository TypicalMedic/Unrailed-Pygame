import pygame as pg
import Sprites as spr
import Entities as ent
from Player import Player
import numpy as np

MAP = [
    ["w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w",
     "w"],
    ["w", "", "t", "", "s", "s", "", "", "", "", "t", "", "", "", "", "", "s", "t", "", "", "", "w", "t", "w"],
    ["w", "", "", "", "", "", "", "", "", "t", "t", "", "", "w", "", "", "t", "", "", "", "", "t", "t", "w"],
    ["w", "t", "t", "t", "", "", "", "", "", "t", "t", "", "", "", "", "", "", "", "", "", "", "s", "s", "w"],
    ["w", "t", "t", "T", "", "", "", "w", "s", "", "", "", "", "t", "", "s", "", "", "", "", "s", "s", "s", "w"],
    ["w", "", "", "p0", "p1", "", "", "s", "s", "", "", "t", "t", "t", "t", "", "", "", "", "", "", "", "", "w"],
    ["w", "", "", "", "", "", "", "w", "", "", "", "t", "t", "t", "t", "", "", "", "", "", "", "", "", "w"],
    ["w", "s", "", "", "", "", "", "", "t", "", "", "", "", "", "", "", "", "", "w", "w", "", "", "", "w"],
    ["w", "s", "s", "", "s", "", "", "", "s", "", "S", "", "", "", "w", "w", "", "", "w", "s", "t", "", "", "w"],
    ["w", "w", "", "t", "s", "", "", "", "s", "", "", "", "", "", "w", "", "", "", "w", "t", "t", "", "", "w"],
    ["w", "", "", "", "s", "s", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "t", "w"],
    ["w", "", "", "w", "w", "w", "", "", "", "", "", "", "", "t", "t", "", "", "", "", "", "", "t", "", "w"],
    ["w", "s", "", "", "", "", "", "", "", "", "", "", "t", "t", "t", "", "", "", "", "", "", "", "", "w"],
    ["w", "", "t", "t", "", "", "", "t", "", "", "", "s", "s", "", "", "", "", "", "", "", "w", "", "", "w"],
    ["w", "", "", "", "", "", "w", "", "", "", "s", "s", "", "", "", "", "t", "", "", "", "", "t", "s", "w"],
    ["w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w",
     "w"]
]


class UnrailedGame:

    def __init__(self, delta_time=30, trees_needed=1, steel_needed=1, square_size=20, info_panel_height=100):

        self.collected_trees = 0
        self.collected_steel = 0
        self.collected_rails = 10
        self.screen_width = 600
        self.screen_height = 500
        self.game_over = False
        self.square_color = 'grey'
        self.bgclr = (255, 255, 255)

        self.p0 = None
        self.p1 = None
        self.rail = None
        self.train = None
        self.station = None
        self.screen = None

        self.rail_list = []
        self.rail_path = []
        self.used_rail_list = []
        self.walls_list = []

        self.trees_list = []
        self.trees_list_sprites = []

        self.steel_list = []
        self.steel_list_sprites = []

        self.MAP_WALLS = []
        self.MAP_STEEL = []
        self.MAP_TREES = []
        self.MAP_PLAYER0 = []
        self.MAP_PLAYER1 = []
        self.MAP_TRAIN = []
        self.MAP_STATION = []
        self.MAP_RAILS = []

        self.tick_amount = delta_time
        self.trees_needed = trees_needed
        self.steel_needed = steel_needed
        self.square_size = square_size
        self.info_panel_height = info_panel_height

        pg.init()
        self.generate_map_from_template(MAP)
        self.rails = pg.sprite.Group(self.rail_list)
        self.used_rail_list.append(self.rail)
        self.used_rails = pg.sprite.Group(self.used_rail_list)
        self.walls = pg.sprite.Group(self.walls_list)
        self.trees = pg.sprite.Group(self.trees_list_sprites)
        self.steel = pg.sprite.Group(self.steel_list_sprites)
        self.dsp = pg.display
        self.clock = pg.time.Clock()
        self.screen.fill(self.bgclr)
        self.draw_grid(self.screen_width, (self.screen_height - self.info_panel_height), self.square_size)
        self.screen.blit(self.p0.image, self.p0.rect)
        self.screen.blit(self.p1.image, self.p1.rect)
        self.screen.blit(self.train.sprite.image, self.train.sprite.rect)
        self.dsp.update()
        self.text_font = pg.font.SysFont("arial", 25)
        self.render_game()

    def render_game(self):
        self.screen.fill(self.bgclr)
        self.draw_grid(self.screen_width, (self.screen_height - self.info_panel_height), self.square_size)
        # pg.draw.rect(self.screen, rect=pl.collider, color='green')
        self.trees.draw(self.screen)
        self.steel.draw(self.screen)
        self.rails.draw(self.screen)
        self.walls.draw(self.screen)
        self.used_rails.draw(self.screen)
        self.screen.blit(self.station.image, self.station.rect)
        self.screen.blit(self.p0.image, self.p0.rect)
        self.screen.blit(self.p1.image, self.p1.rect)
        self.screen.blit(self.train.sprite.image, self.train.sprite.rect)
        self.draw_game_info()
        pg.display.update()

    def step(self):
        if pg.time.get_ticks() > self.train.delay:
            self.train.move_train(self.tick_amount, self.rail_list, self.used_rail_list, self.station)
            if self.train.sprite.rect.collidelist(self.rail_list) == -1 and \
                    self.train.sprite.rect.collidelist(self.used_rail_list) == -1:
                print('game over')
                self.game_over = True
            if self.train.sprite.rect.collidelist(self.rail_list) != -1:
                self.train.speed += self.train.speed * 0.1
                ur = self.rail_list.pop(self.train.sprite.rect.collidelist(self.rail_list))
                ur.image = pg.image.load('Assets/usedrail.png').convert_alpha()
                self.used_rail_list.append(ur)
                self.used_rails = pg.sprite.Group(self.used_rail_list)
                self.rails = pg.sprite.Group(self.rail_list)
            if self.train.sprite.rect.colliderect(self.station):
                print('game won!')
                self.game_over = True
            # collide_trees = pl.collider.collidelist(self.trees_list_sprites)
            # collide_steel = pl.collider.collidelist(self.steel_list_sprites)
            # collide_train = pl.collider.colliderect(self.train.sprite)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.game_over = True
        self.clock.tick(self.tick_amount)
        return self.game_over

    def run_game_singleplayer(self):
        while not self.game_over:
            self.step()
            is_dir_key_pressed = False
            collide_trees = self.p0.collider.collidelist(self.trees_list_sprites)
            collide_steel = self.p0.collider.collidelist(self.steel_list_sprites)
            collide_train = self.p0.collider.colliderect(self.train.sprite)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.game_over = True
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_LEFT:
                        self.p0.pl_dir = self.p0.direction['left']
                        is_dir_key_pressed = True
                    elif event.key == pg.K_RIGHT:
                        self.p0.pl_dir = self.p0.direction['right']
                        is_dir_key_pressed = True
                    elif event.key == pg.K_UP:
                        self.p0.pl_dir = self.p0.direction['up']
                        is_dir_key_pressed = True
                    elif event.key == pg.K_DOWN:
                        self.p0.pl_dir = self.p0.direction['down']
                        is_dir_key_pressed = True
                    self.p0.set_dir()
                    collide_trees = self.p0.collider.collidelist(self.trees_list_sprites)
                    collide_steel = self.p0.collider.collidelist(self.steel_list_sprites)
                    collide_train = self.p0.collider.colliderect(self.train.sprite)
                    collide_station = self.p0.collider.colliderect(self.station)
                    collide_walls = self.p0.collider.collidelist(self.walls_list)
                    if collide_trees == -1 and collide_steel == -1 and not collide_train \
                            and not collide_station and collide_walls == -1:
                        if is_dir_key_pressed:
                            self.p0.set_pos()
                    if event.key == pg.K_SPACE:
                        if self.p0.collider.collidelist(self.rail_list) == -1:
                            if self.collected_rails >= 1 and collide_steel == -1 \
                                    and collide_trees == -1 and collide_walls == -1 \
                                    and self.p0.collider.collidelist(self.used_rail_list) == -1:
                                self.collected_rails -= 1
                                rail = spr.sprite(self.p0.collider.x + self.square_size / 2,
                                self.p0.collider.y + self.square_size / 2,
                                                  'Assets/rail.png')
                                self.rail_list.append(rail)
                                self.rails = pg.sprite.Group(self.rail_list)
                        else:
                            self.collected_rails += 1
                            self.rail_list.pop(self.p0.collider.collidelist(self.rail_list))
                            self.rails = pg.sprite.Group(self.rail_list)

                    # чисто чтобы смотреть где коллайдер, потом удалить?
                    self.p0.set_dir()
            if collide_train:
                if self.collected_trees >= self.trees_needed and self.collected_steel >= self.steel_needed:
                    self.collected_trees -= self.trees_needed
                    self.collected_steel -= self.steel_needed
                    self.collected_rails += 1
            if collide_trees != -1:
                if self.trees_list[collide_trees].damage():
                    self.collected_trees += self.trees_list[collide_trees].loot
                    self.trees_list.pop(collide_trees)
                    self.trees_list_sprites = self.get_entities_sprites(self.trees_list)
                    self.trees = pg.sprite.Group(self.trees_list_sprites)
            if collide_steel != -1:
                if self.steel_list[collide_steel].damage():
                    self.collected_steel += self.steel_list[collide_steel].loot
                    self.steel_list.pop(collide_steel)
                    self.steel_list_sprites = self.get_entities_sprites(self.steel_list)
                    self.steel = pg.sprite.Group(self.steel_list_sprites)

            self.render_game()

        self.dsp.quit()

    def generate_map_random(self):
        return

    # optimize!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def generate_map_from_template(self, map_template: [str, str]):
        self.MAP_WALLS = np.zeros((len(map_template), len(map_template[0])))
        self.MAP_STEEL = np.zeros((len(map_template), len(map_template[0])))
        self.MAP_TREES = np.zeros((len(map_template), len(map_template[0])))
        self.MAP_PLAYER0 = np.zeros((len(map_template), len(map_template[0])))
        self.MAP_PLAYER1 = np.zeros((len(map_template), len(map_template[0])))
        self.MAP_TRAIN = np.zeros((len(map_template), len(map_template[0])))
        self.MAP_STATION = np.zeros((len(map_template), len(map_template[0])))
        self.MAP_RAILS = np.zeros((len(map_template), len(map_template[0])))

        self.screen_width = len(map_template[0]) * self.square_size
        self.screen_height = len(map_template) * self.square_size + self.info_panel_height
        self.screen = pg.display.set_mode(size=[self.screen_width, self.screen_height])
        for y in range(len(map_template)):
            for x in range(len(map_template[y])):
                if map_template[y][x] == "w":
                    self.MAP_WALLS[y][x] = 1
                    self.walls_list.append(
                        spr.sprite(x * self.square_size + self.square_size / 2,
                                   y * self.square_size + self.square_size / 2, 'Assets/wall.png'))
                elif map_template[y][x] == "t":
                    self.MAP_TREES[y][x] = 1
                    self.trees_list.append(
                        ent.Resource(spr.sprite(x * self.square_size + self.square_size / 2,
                                                y * self.square_size + self.square_size / 2, 'Assets/tree.png'), 10, 1))
                elif map_template[y][x] == "s":
                    self.MAP_STEEL[y][x] = 1
                    self.steel_list.append(
                        ent.Resource(spr.sprite(x * self.square_size + self.square_size / 2,
                                                y * self.square_size + self.square_size / 2, 'Assets/steel.png'), 10,
                                     1))
                elif map_template[y][x] == "p0":
                    self.MAP_PLAYER0[y][x] = 1
                    self.p0 = Player(self.square_size, self.square_size, self.p1, x, y, self.square_size)
                elif map_template[y][x] == "p1":
                    self.MAP_PLAYER1[y][x] = 1
                    self.p1 = Player(self.square_size, self.square_size, self.p0, x, y, self.square_size)
                elif map_template[y][x] == "S":
                    self.MAP_STATION[y][x] = 1
                    self.station = spr.sprite(x * self.square_size + self.square_size / 2,
                                              y * self.square_size + self.square_size / 2, 'Assets/station.png')
                elif map_template[y][x] == "T":
                    self.MAP_TRAIN[y][x] = 1
                    self.MAP_RAILS[y][x] = 1
                    self.train = ent.train(x * self.square_size + self.square_size / 2,
                                           y * self.square_size + self.square_size / 2, self.square_size,
                                           'Assets/train.png')
                    self.rail = spr.sprite(x * self.square_size + self.square_size / 2,
                                           y * self.square_size + self.square_size / 2, 'Assets/usedrail.png')
                else:
                    continue
        self.steel_list_sprites = self.get_entities_sprites(self.steel_list)
        self.trees_list_sprites = self.get_entities_sprites(self.trees_list)
        return

    def draw_grid(self, w, h, size):
        """
        draws black grid
        Parameters
        ----------
        w
            screen width in pixels
        h
            screen height in pixels
        size: int
            cell size in pixels
        """
        for i in range(h // size):
            for j in range(w // size):
                pg.draw.rect(self.screen, color=self.square_color,
                             rect=[size * j, size * i, size, size], width=1)

    @staticmethod
    def get_entities_sprites(entities: []):
        """
        :returns:
        array of sprites from entities
        Parameters
        ----------
        entities:
        array of entity type
        """
        sprt = []
        for i in entities:
            sprt.append(i.sprite)
        return sprt

    def draw_game_info(self):
        """
        draws UI on the bottom of the game
        """
        msg = 'trees: ' + str(self.collected_trees)
        msg += '   steel: ' + str(self.collected_steel)
        msg += '   rails: ' + str(self.collected_rails)
        msg1 = 'train speed: '
        if pg.time.get_ticks() < self.train.start_delay:
            msg += '    train departs in: ' + str((self.train.start_delay - pg.time.get_ticks()) // 1000)
            msg1 += '0'
        else:
            msg1 += str(self.train.speed)
        info_text = self.text_font.render(msg, True, 'black')
        self.screen.blit(info_text, [10, self.screen.get_height() - self.info_panel_height * 9 / 10])
        info_text = self.text_font.render(msg1, True, 'black')
        self.screen.blit(info_text, [10, self.screen.get_height() - self.info_panel_height * 6 / 10])

    def env_action_update(self, action, player: Player):
        interact = False
        if action == 0:
            player.pl_dir = player.direction['left']
        elif action == 1:
            player.pl_dir = player.direction['right']
        elif action == 2:
            player.pl_dir = player.direction['up']
        elif action == 3:
            player.pl_dir = player.direction['down']
        elif action == 4:
            pass
        elif action == 5:
            interact = True
        player.set_dir()
        collide_trees = player.collider.collidelist(self.trees_list_sprites)
        collide_steel = player.collider.collidelist(self.steel_list_sprites)
        collide_train = player.collider.colliderect(self.train.sprite)
        collide_station = player.collider.colliderect(self.station)
        collide_walls = player.collider.collidelist(self.walls_list)
        if collide_trees == -1 and collide_steel == -1 and not collide_train \
                and not collide_station and collide_walls == -1:
            if 0 <= action < 4:
                player.set_pos()
        if interact:
            if player.collider.collidelist(self.rail_list) == -1:
                if self.collected_rails >= 1 and collide_steel == -1 \
                        and collide_trees == -1 and collide_walls == -1 \
                        and player.collider.collidelist(self.used_rail_list) == -1:
                    self.collected_rails -= 1
                    rail = spr.sprite(player.collider.x + self.square_size / 2,
                                      player.collider.y + self.square_size / 2,
                                      'Assets/rail.png')
                    self.rail_list.append(rail)
                    self.rails = pg.sprite.Group(self.rail_list)
            else:
                self.collected_rails += 1
                self.rail_list.pop(player.collider.collidelist(self.rail_list))
                self.rails = pg.sprite.Group(self.rail_list)

        if collide_train:
            if self.collected_trees >= self.trees_needed and self.collected_steel >= self.steel_needed:
                self.collected_trees -= self.trees_needed
                self.collected_steel -= self.steel_needed
                self.collected_rails += 1
        if collide_trees != -1:
            if self.trees_list[collide_trees].damage():
                self.collected_trees += self.trees_list[collide_trees].loot
                self.trees_list.pop(collide_trees)
                self.trees_list_sprites = self.get_entities_sprites(self.trees_list)
                self.trees = pg.sprite.Group(self.trees_list_sprites)
        if collide_steel != -1:
            if self.steel_list[collide_steel].damage():
                self.collected_steel += self.steel_list[collide_steel].loot
                self.steel_list.pop(collide_steel)
                self.steel_list_sprites = self.get_entities_sprites(self.steel_list)
                self.steel = pg.sprite.Group(self.steel_list_sprites)
