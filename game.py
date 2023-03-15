import math

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
    ["w", "t", "t", "t", "", "", "", "t", "", "t", "t", "", "", "", "", "", "", "", "", "", "", "s", "s", "w"],
    ["w", "t", "t", "T", "", "", "", "w", "s", "", "", "", "", "t", "", "s", "", "", "", "", "s", "s", "s", "w"],
    ["w", "", "p0", "", "p1", "", "", "s", "s", "", "", "t", "t", "t", "t", "", "", "", "", "", "", "", "", "w"],
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

    def __init__(self, delta_time=1, trees_needed=1, steel_needed=1, square_size=20, info_panel_height=100):

        self.tick_count = 0
        self.collected_trees = 0
        self.collected_steel = 0
        self.collected_rails = 10
        self.screen_width = 600
        self.screen_height = 500
        self.game_over = False
        self.square_color = 'grey'
        self.bgclr = (255, 255, 255)
        self.rail_path_completed = False

        self.p0 = None
        self.p1 = None
        self.rail = None
        self.train = None
        self.station = None
        self.screen = None

        self.rail_list = []
        self.rail_path = []
        self.used_rail_list = []
        self.last_rail_colliders = []
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
        self.agents = {"player_0": self.p0, "player_1": self.p1}
        self.rails = pg.sprite.Group(self.rail_list)
        self.used_rail_list.append(self.rail)
        self.used_rails = pg.sprite.Group(self.used_rail_list)
        self.rail_paths = pg.sprite.Group(self.rail_path)
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
        self.render_game([])

    def render_game(self, agent_rewards: []):
        self.screen.fill(self.bgclr)
        self.draw_grid(self.screen_width, (self.screen_height - self.info_panel_height), self.square_size)
        # pg.draw.rect(self.screen, rect=pl.collider, color='green')
        self.trees.draw(self.screen)
        self.steel.draw(self.screen)
        self.rails.draw(self.screen)
        self.walls.draw(self.screen)
        self.rail_paths.draw(self.screen)
        self.used_rails.draw(self.screen)
        self.screen.blit(self.station.image, self.station.rect)
        self.screen.blit(self.p0.image, self.p0.rect)
        self.screen.blit(self.p1.image, self.p1.rect)
        self.screen.blit(self.train.sprite.image, self.train.sprite.rect)
        self.draw_game_info(agent_rewards)
        # pg.draw.rect(self.screen, color="blue", rect=self.last_rail_colliders[0])
        # pg.draw.rect(self.screen, color="blue", rect=self.last_rail_colliders[1])
        # pg.draw.rect(self.screen, color="blue", rect=self.last_rail_colliders[2])
        # pg.draw.rect(self.screen, color="blue", rect=self.last_rail_colliders[3])

        pg.display.update()

    def step(self):
        reward = 0
        if self.tick_count > self.train.delay:
            found_path, found_station = self.train.move_train(self.tick_amount, self.rail_path,
                                                              self.used_rail_list, self.station)
            if found_path:
                # self.train.speed += self.train.speed * 0.1
                ur = self.rail_path.pop(0)
                ur.image = pg.image.load('Assets/usedrail.png').convert_alpha()
                x = self.used_rail_list[-1].rect.x // self.square_size
                y = self.used_rail_list[-1].rect.y // self.square_size
                # to_station_before = math.dist([x, y], [self.station_x, self.station_y])
                self.MAP_TRAIN[y][x] = 0
                self.used_rail_list.append(ur)
                x = self.used_rail_list[-1].rect.x // self.square_size
                y = self.used_rail_list[-1].rect.y // self.square_size
                # to_station_after = math.dist([x, y], [self.station_x, self.station_y])
                # if to_station_before > to_station_after:
                #     reward += 25  # change to fully depend on the distance
                # else:
                #     reward -= 10
                self.MAP_TRAIN[y][x] = 1
                self.MAP_RAILS[y][x] = 3
                self.train.x = x
                self.train.y = y
                self.used_rails = pg.sprite.Group(self.used_rail_list)
                self.rail_paths = pg.sprite.Group(self.rail_path)
            if found_station:
                print('game won!')
                reward = 2e4/len(self.used_rail_list) + 500
                self.game_over = True
            elif not self.train.tr_dir["col"].colliderect(self.used_rail_list[-1]):
                # print('game over')
                reward = -1e7/(self.tick_count + 1) - 500
                self.game_over = True
            # collide_trees = pl.collider.collidelist(self.trees_list_sprites)
            # collide_steel = pl.collider.collidelist(self.steel_list_sprites)
            # collide_train = pl.collider.colliderect(self.train.sprite)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                reward = 0
                self.game_over = True
        # self.clock.tick(self.tick_amount)
        # self.tick_count += 1
        return self.game_over, reward

    def run_game_singleplayer(self):
        while not self.game_over:
            self.step()
            self.tick_count += 1
            self.clock.tick(self.tick_amount)
            is_dir_key_pressed = False
            collide_trees = self.p0.collider.collidelist(self.trees_list_sprites)
            collide_steel = self.p0.collider.collidelist(self.steel_list_sprites)
            collide_train = self.p0.collider.colliderect(self.train.sprite)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.game_over = True
                if event.type == pg.KEYDOWN and not self.rail_path_completed:
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
                        if self.p0.collider.collidelist(self.rail_list) != -1:
                            self.collected_rails += 1
                            self.rail_list.pop(self.p0.collider.collidelist(self.rail_list))
                            self.rails = pg.sprite.Group(self.rail_list)
                        elif self.p0.collider.collidelist(self.rail_path) != -1:
                            if self.p0.collider.collidelist(self.rail_path) == len(self.rail_path) - 1:
                                self.collected_rails += 1
                                self.rail_path.pop(self.p0.collider.collidelist(self.rail_path))
                                self.rail_paths = pg.sprite.Group(self.rail_path)

                                if len(self.rail_path) == 0:
                                    x = self.used_rail_list[-1].rect.x
                                    y = self.used_rail_list[-1].rect.y
                                else:
                                    x = self.rail_path[-1].rect.x
                                    y = self.rail_path[-1].rect.y
                                self.last_rail_colliders = [
                                    pg.rect.Rect(x, y - self.square_size, self.square_size, self.square_size),  # up
                                    pg.rect.Rect(x, y + self.square_size, self.square_size, self.square_size),  # down
                                    pg.rect.Rect(x + self.square_size, y, self.square_size, self.square_size),  # right
                                    pg.rect.Rect(x - self.square_size, y, self.square_size, self.square_size)]  # left
                                self.calculate_rail_path()
                        else:
                            if self.collected_rails >= 1 and collide_steel == -1 \
                                    and collide_trees == -1 and collide_walls == -1 \
                                    and self.p0.collider.collidelist(self.used_rail_list) == -1:
                                self.collected_rails -= 1
                                rail = spr.sprite(self.p0.collider.x + self.square_size / 2,
                                                  self.p0.collider.y + self.square_size / 2,
                                                  'Assets/rail.png')
                                self.rail_list.append(rail)
                                self.rails = pg.sprite.Group(self.rail_list)
                                self.calculate_rail_path()

                                # while len(self.rail_path) != 0:
                                #     ur = self.rail_path.pop(-1)
                                #     ur.image = pg.image.load('Assets/usedrail.png').convert_alpha()
                                #     self.used_rail_list.append(ur)
                                # self.used_rails = pg.sprite.Group(self.used_rail_list)
                                # self.rail_paths = pg.sprite.Group(self.rail_path)
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

            self.render_game([0])

        self.dsp.quit()

    def calculate_rail_path(self):
        reward = 0
        clear = False
        while not clear:
            clear = True
            for col in self.last_rail_colliders:
                if col.collidelist(self.rail_list) != -1:
                    clear = False
                    x = self.rail_path[-1].rect.x // self.square_size if len(self.rail_path) != 0 else self.used_rail_list[-1].rect.x // self.square_size
                    y = self.rail_path[-1].rect.y // self.square_size if len(self.rail_path) != 0 else self.used_rail_list[-1].rect.y // self.square_size
                    to_station_before = math.dist([x, y], [self.station_x, self.station_y])
                    x = self.rail_list[col.collidelist(self.rail_list)].rect.x
                    y = self.rail_list[col.collidelist(self.rail_list)].rect.y
                    self.last_rail_colliders = [
                        pg.rect.Rect(x, y - self.square_size, self.square_size, self.square_size),  # up
                        pg.rect.Rect(x, y + self.square_size, self.square_size, self.square_size),  # down
                        pg.rect.Rect(x + self.square_size, y, self.square_size, self.square_size),  # right
                        pg.rect.Rect(x - self.square_size, y, self.square_size, self.square_size)]  # left
                    self.rail_path.append(self.rail_list.pop(col.collidelist(self.rail_list)))
                    x = self.rail_path[-1].rect.x // self.square_size
                    y = self.rail_path[-1].rect.y // self.square_size
                    self.MAP_RAILS[y][x] = 2
                    to_station_after = math.dist([x, y], [self.station_x, self.station_y])
                    if to_station_before > to_station_after:
                        reward += 1000/(to_station_after + 1)
                    else:
                        reward -= 10 * to_station_after
                    self.rail_path[-1].image = pg.image.load("Assets/rail_path.png").convert_alpha()
                    self.rail_paths = pg.sprite.Group(self.rail_path)
                    self.rails = pg.sprite.Group(self.rail_list)
                    break
                elif col.colliderect(self.station):
                    self.train.speed /= 1000
                    self.rail_path_completed = True
                    for r in self.rail_path:
                        r.image = pg.image.load('Assets/usedrail.png').convert_alpha()
        return reward

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

        self.field_x = len(map_template[0])
        self.field_y = len(map_template)
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
                    self.station_x = x
                    self.station_y = y
                    self.station = spr.sprite(x * self.square_size + self.square_size / 2,
                                              y * self.square_size + self.square_size / 2, 'Assets/station.png')
                elif map_template[y][x] == "T":
                    self.MAP_TRAIN[y][x] = 1
                    self.MAP_RAILS[y][x] = 3    # 1: bad rails 2: path rails 3: used rails
                    self.train = ent.Train(x * self.square_size + self.square_size / 2,
                                           y * self.square_size + self.square_size / 2, self.square_size,
                                           'Assets/train.png', 0)
                    self.train.x = x
                    self.train.y = y
                    self.rail = spr.sprite(x * self.square_size + self.square_size / 2,
                                           y * self.square_size + self.square_size / 2, 'Assets/usedrail.png')
                    self.last_rail_colliders = [
                        pg.rect.Rect(x * self.square_size, y * self.square_size - self.square_size, self.square_size,
                                     self.square_size),  # up
                        pg.rect.Rect(x * self.square_size, y * self.square_size + self.square_size, self.square_size,
                                     self.square_size),  # down
                        pg.rect.Rect(x * self.square_size + self.square_size, y * self.square_size, self.square_size,
                                     self.square_size),  # right
                        pg.rect.Rect(x * self.square_size - self.square_size, y * self.square_size, self.square_size,
                                     self.square_size)]  # left
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

    def draw_game_info(self, agent_rewards: []):
        """
        draws UI on the bottom of the game
        """
        msg = 'trees: ' + str(self.collected_trees)
        msg += '   steel: ' + str(self.collected_steel)
        msg += '   rails: ' + str(self.collected_rails)
        msg1 = 'train speed: '
        if self.tick_count < self.train.start_delay:
            msg += '    train departs in: ' + str((self.train.delay - self.tick_count))
            msg1 += '0'
        else:
            msg1 += str(self.train.speed)
        msg1 += '   rewards: '
        for i in agent_rewards:
            msg1 += '%.2f, ' % i
        info_text = self.text_font.render(msg, True, 'black')
        self.screen.blit(info_text, [10, self.screen.get_height() - self.info_panel_height * 9 / 10])
        info_text = self.text_font.render(msg1, True, 'black')
        self.screen.blit(info_text, [10, self.screen.get_height() - self.info_panel_height * 6 / 10])

    def env_action_update(self, action, player: Player, map_pl):
        reward = 0
        if action is None:
            return reward
        PLACE_RAIL = False
        REMOVE_RAIL = False
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
            PLACE_RAIL = True
        elif action == 6:
            REMOVE_RAIL = True
        player.set_dir()
        collide_trees = player.collider.collidelist(self.trees_list_sprites)
        collide_steel = player.collider.collidelist(self.steel_list_sprites)
        collide_train = player.collider.colliderect(self.train.sprite)
        collide_station = player.collider.colliderect(self.station)
        collide_walls = player.collider.collidelist(self.walls_list)
        if collide_trees == -1 and collide_steel == -1 and not collide_train \
                and not collide_station and collide_walls == -1:
            if 0 <= action < 4:
                map_pl[player.y][player.x] = 0
                player.set_pos()
                map_pl[player.y][player.x] = 1
        if 0 <= action <= 3 and (collide_station or collide_walls != -1):
            reward -= 15
        x = player.collider.x // self.square_size
        y = player.collider.y // self.square_size
        if PLACE_RAIL:
            if player.collider.collidelist(self.rail_list) == -1 and player.collider.collidelist(self.rail_path) == -1:
                if self.collected_rails >= 1 and collide_steel == -1 \
                        and collide_trees == -1 and collide_walls == -1 \
                        and player.collider.collidelist(self.used_rail_list) == -1:
                    placed_bad_rails_len_before = len(self.rail_list)
                    self.collected_rails -= 1
                    self.MAP_RAILS[y][x] = 1
                    rail = spr.sprite(player.collider.x + self.square_size / 2,
                                      player.collider.y + self.square_size / 2,
                                      'Assets/rail.png')
                    self.rail_list.append(rail)
                    self.rails = pg.sprite.Group(self.rail_list)

                    reward += self.calculate_rail_path()
                    if len(self.rail_list) > placed_bad_rails_len_before:
                        reward -= 30 * math.dist([x, y], [self.station_x, self.station_y])
                else:
                    reward -= 10
            else:
                reward -= 10
        if REMOVE_RAIL:
            if player.collider.collidelist(self.rail_list) != -1:
                reward += 15
                self.collected_rails += 1
                self.rail_list.pop(player.collider.collidelist(self.rail_list))
                self.MAP_RAILS[y][x] = 0
                self.rails = pg.sprite.Group(self.rail_list)
            elif player.collider.collidelist(self.rail_path) != -1:
                if player.collider.collidelist(self.rail_path) == len(self.rail_path) - 1:
                    self.collected_rails += 1
                    self.MAP_RAILS[y][x] = 0
                    self.rail_path.pop(player.collider.collidelist(self.rail_path))
                    self.rail_paths = pg.sprite.Group(self.rail_path)
                    if len(self.rail_path) == 0:
                        x = self.used_rail_list[-1].rect.x
                        y = self.used_rail_list[-1].rect.y
                    else:
                        x = self.rail_path[-1].rect.x
                        y = self.rail_path[-1].rect.y

                    to_station = math.dist([x, y], [self.station_x, self.station_y])
                    reward -= 12 * to_station
                    self.last_rail_colliders = [
                        pg.rect.Rect(x, y - self.square_size, self.square_size, self.square_size),  # up
                        pg.rect.Rect(x, y + self.square_size, self.square_size, self.square_size),  # down
                        pg.rect.Rect(x + self.square_size, y, self.square_size, self.square_size),  # right
                        pg.rect.Rect(x - self.square_size, y, self.square_size, self.square_size)]  # left
                    reward += self.calculate_rail_path()
            else:
                reward -= 10

        if collide_train:
            reward += 10
            if self.collected_trees >= self.trees_needed and self.collected_steel >= self.steel_needed:
                self.collected_trees -= self.trees_needed
                self.collected_steel -= self.steel_needed
                self.collected_rails += 1
                reward += 250
        if collide_trees != -1:
            reward += 5
            if self.trees_list[collide_trees].damage():
                reward += 15
                if self.collected_trees < self.collected_steel:
                    reward += self.trees_list[collide_trees].loot * 100
                self.collected_trees += self.trees_list[collide_trees].loot
                self.trees_list.pop(collide_trees)
                self.MAP_TREES[y][x] = 0
                self.trees_list_sprites = self.get_entities_sprites(self.trees_list)
                self.trees = pg.sprite.Group(self.trees_list_sprites)
        if collide_steel != -1:
            reward += 5
            if self.steel_list[collide_steel].damage():
                reward += 15
                if self.collected_trees > self.collected_steel:
                    reward += self.steel_list[collide_steel].loot * 100
                self.collected_steel += self.steel_list[collide_steel].loot
                self.steel_list.pop(collide_steel)
                self.MAP_STEEL[y][x] = 0
                self.steel_list_sprites = self.get_entities_sprites(self.steel_list)
                self.steel = pg.sprite.Group(self.steel_list_sprites)
        return reward
