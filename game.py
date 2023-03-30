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
    ["w", "t", "t", "T", "r", "r", "r", "w", "s", "", "", "", "", "t", "", "s", "", "", "", "", "s", "s", "s", "w"],
    ["w", "", "", "", "", "", "", "s", "s", "", "", "t", "t", "t", "t", "", "", "", "", "", "", "", "", "w"],
    ["w", "", "p0", "", "p1", "", "", "w", "", "", "", "t", "t", "t", "t", "", "", "", "", "", "", "", "", "w"],
    ["w", "s", "", "", "", "", "", "", "t", "", "", "", "", "", "", "", "", "", "w", "w", "", "", "", "w"],
    ["w", "s", "s", "", "s", "", "", "", "s", "", "S", "", "", "", "w", "w", "", "", "w", "s", "t", "", "", "w"],
    ["w", "w", "", "t", "s", "", "", "", "s", "", "", "", "", "", "w", "", "", "", "w", "t", "t", "", "", "w"],
    ["w", "", "", "", "s", "s", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "t", "w"],
    ["w", "", "", "w", "w", "w", "", "", "", "", "", "", "", "t", "t", "", "", "", "", "", "", "", "t", "w"],
    ["w", "s", "", "", "", "", "", "", "", "", "", "", "t", "t", "t", "", "", "", "", "", "", "", "", "w"],
    ["w", "", "t", "t", "", "", "", "t", "", "", "", "s", "s", "", "", "", "", "", "", "", "w", "", "", "w"],
    ["w", "", "", "", "", "", "w", "", "", "", "s", "s", "", "", "", "", "t", "", "", "", "", "t", "s", "w"],
    ["w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w",
     "w"]
]

REWARDS = {"win": 10000,
           "lose": -1000,
           "passed 1 rail": 200,
           "put path closer": 1000,
           "put path further": 250,
           "every step": -5,
           "bad rail placed": -50,
           "bad rail removed": 10,
           "path removed": -2000,
           "made rail": 50,
           "tree/steel collided": 10,
           "got tree/steel": 60,
           "bonus": 10
           }


class UnrailedGame:
    """
    Unrailed game simplified replica using PyGame
    """

    def __init__(self, delta_time=1, trees_needed=1, steel_needed=1, square_size=20, info_panel_height=100):
        """
        Initializes main game elements and renders first frame.

        Parameters:
            :param delta_time: how much frames pass after one tick, 1 as the default
            :param trees_needed: how much tree resource is needed to craft one rail, 1 as the default
            :param steel_needed: how much steel resource is needed to craft one rail, 1 as the default
            :param square_size: size of one grid square in pixels, 20 as the default
            :param info_panel_height: height of the info panel in pixels, 100 as the default
        """
        self.episode = 0    # current training episode
        self.tick_count = 0
        self.collected_trees = 0
        self.collected_steel = 0
        self.collected_rails = 10
        self.screen_width = 600
        self.screen_height = 500
        self.game_over = False
        self.game_won = False
        self.square_color = 'light grey'
        self.bgclr = (255, 255, 255)
        self.rail_path_completed = False

        self.p0 = None  # first agent
        self.p1 = None  # second agent
        self.rail = None
        self.train = None
        self.station = None
        self.screen = None  # PyGame screen

        self.rail_list = []
        self.rail_path = []
        self.used_rail_list = []
        self.last_rail_colliders = []   # colliders of last rail in the path
        self.walls_list = []

        self.trees_list = []
        self.trees_list_sprites = []    # sprite group for rendering

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

        pg.init()   # initializing game
        self.generate_map_from_template(MAP)    # making game field
        self.p0.ally = self.p1
        self.p1.ally = self.p0
        self.agents = {"player_0": self.p0, "player_1": self.p1}
        self.rails = pg.sprite.Group(self.rail_list)
        self.used_rail_list.append(self.rail)   # self.rail is the one under the train, so it goes to the used rail list
        self.used_rails = pg.sprite.Group(self.used_rail_list)
        self.rail_paths = pg.sprite.Group(self.rail_path)
        self.walls = pg.sprite.Group(self.walls_list)
        self.trees = pg.sprite.Group(self.trees_list_sprites)
        self.steel = pg.sprite.Group(self.steel_list_sprites)
        self.clock = pg.time.Clock()    # used to control time passing in-game
        self.dsp = pg.display
        self.text_font = pg.font.SysFont("arial", 25)
        self.calculate_rail_path()  # create (if possible) rail path after creating game field

    def render_game(self, agent_rewards: []):
        """
        Renders one game frame using PyGamE

        :param agent_rewards: Array of current agents rewards
        """
        self.screen.fill(self.bgclr)
        # renders grid on the field
        self.draw_grid(self.screen_width, (self.screen_height - self.info_panel_height), self.square_size)
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
        self.draw_game_info(agent_rewards)  # render game info in the bottom
        pg.display.update()     # update render changes

    def step(self):
        """
        step function is called at the start every frame. It moves train forward and checks whether game was won or lost

        :returns: whether the game has ended, hot it ended (win/loss)
            and collected reward after step (for the AI training)
        """
        reward = 0
        if self.tick_count > self.train.delay:
            found_path, found_station = self.train.move_train(self.rail_path, self.used_rail_list, self.station)
            if found_path:   # when train reaches rail, being part oh the path
                ur = self.rail_path.pop(0)
                ur.image = pg.image.load('Assets/usedrail.png').convert_alpha()
                # calculate grid x for previous train position, we use last used rail because train just passed it
                x = self.used_rail_list[-1].rect.x // self.square_size
                y = self.used_rail_list[-1].rect.y // self.square_size
                self.MAP_TRAIN[y][x] = 0    # train moved from one cell to another, so we update map correspondingly
                self.used_rail_list.append(ur)
                x = self.used_rail_list[-1].rect.x // self.square_size  # new train position in grid
                y = self.used_rail_list[-1].rect.y // self.square_size
                self.MAP_TRAIN[y][x] = 1
                self.MAP_RAILS[y][x] = 3    # mark current rail under the train as used
                self.train.x = x
                self.train.y = y
                self.used_rails = pg.sprite.Group(self.used_rail_list)
                self.rail_paths = pg.sprite.Group(self.rail_path)
            if found_station:   # when train reaches station
                print('game won!')
                reward = REWARDS["win"]
                self.game_over = True
                self.game_won = True
            # when train has no rail under its sprite even a little
            elif not self.train.tr_dir["col"].colliderect(self.used_rail_list[-1]):
                reward = REWARDS["lose"] + REWARDS["passed 1 rail"] * len(self.used_rail_list) ** 1.1
                self.game_over = True
        for event in pg.event.get():    # when window is closed manually
            if event.type == pg.QUIT:
                reward = 0
                self.game_over = True
        return self.game_over, self.game_won, reward

    def run_game_singleplayer(self):
        """
        Launches the game in singleplyer. Loops over until the game is over, reads user input and calculates collisions
        INPUTS:
            up, down, left, right - keyboard up, down, left, right
            place, remove rails - space
        """
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
                        if self.p0.rect.collidelist(self.rail_list) != -1:
                            self.collected_rails += 1
                            self.rail_list.pop(self.p0.rect.collidelist(self.rail_list))
                            self.rails = pg.sprite.Group(self.rail_list)
                        elif len(self.rail_path) > 0 and self.p0.rect.colliderect(self.rail_path[-1]):
                            self.collected_rails += 1
                            self.rail_path.pop(self.p0.rect.collidelist(self.rail_path))
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
                            if self.collected_rails >= 1 and self.p0.rect.collidelist(self.used_rail_list) == -1:
                                self.collected_rails -= 1
                                rail = spr.CustomSprite(self.p0.rect.x + self.square_size / 2,
                                                        self.p0.rect.y + self.square_size / 2,
                                                  'Assets/rail.png')
                                self.rail_list.append(rail)
                                self.rails = pg.sprite.Group(self.rail_list)
                                self.calculate_rail_path()
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
        """
        builds rail path if there is one
        :return: collected reward (for the AI training)
        """
        reward = 0
        clear = False
        while not clear:    # loops until every possible rail for path is found
            clear = True
            for col in self.last_rail_colliders:    # check upper, lower, left and right cells for unused rails
                if col.collidelist(self.rail_list) != -1:   # found unused rail
                    clear = False   # now we have to re-check to ensure no more unused rails exist near new path's end
                    # update maps accordingly
                    if len(self.rail_path) != 0:
                        x = self.rail_path[-1].rect.x // self.square_size
                        y = self.rail_path[-1].rect.y // self.square_size
                        self.MAP_RAILS[y][x] = 2    # if unwalked path is not empty
                    else:
                        x = self.used_rail_list[-1].rect.x // self.square_size
                        y = self.used_rail_list[-1].rect.y // self.square_size
                        self.MAP_RAILS[y][x] = 3    # if unwalked path is empty (last rail in path was already used)
                    # distance from path's end to the station before path extending
                    to_station_before = math.dist([x, y], [self.station_x, self.station_y])
                    # new path's end coordinates
                    x = self.rail_list[col.collidelist(self.rail_list)].rect.x
                    y = self.rail_list[col.collidelist(self.rail_list)].rect.y
                    # move colliders to the right place
                    self.last_rail_colliders = [
                        pg.rect.Rect(x, y - self.square_size, self.square_size, self.square_size),  # up
                        pg.rect.Rect(x, y + self.square_size, self.square_size, self.square_size),  # down
                        pg.rect.Rect(x + self.square_size, y, self.square_size, self.square_size),  # right
                        pg.rect.Rect(x - self.square_size, y, self.square_size, self.square_size)]  # left
                    self.rail_path.append(self.rail_list.pop(col.collidelist(self.rail_list)))
                    x = self.rail_path[-1].rect.x // self.square_size
                    y = self.rail_path[-1].rect.y // self.square_size
                    self.MAP_RAILS[y][x] = 4    # mark this cell as the last rail in the path
                    to_station_after = math.dist([x, y], [self.station_x, self.station_y])
                    # reward differs based on distance before and after
                    if to_station_before > to_station_after:
                        reward += REWARDS["put path closer"]
                    else:
                        reward += REWARDS["put path further"]
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

    def generate_map_from_template(self, map_template: [str, str]):
        """
        places game objects from map_template to the right place for further render

        :param map_template: array of the map
        """
        width = len(map_template[0])
        height = len(map_template)
        map = np.zeros((height, width))
        # region Map initialization
        self.MAP_WALLS = map.copy()
        self.MAP_STEEL = map.copy()
        self.MAP_TREES = map.copy()
        self.MAP_PLAYER0 = map.copy()
        self.MAP_PLAYER1 = map.copy()
        self.MAP_TRAIN = map.copy()
        self.MAP_STATION = map.copy()
        self.MAP_RAILS = map.copy()
        # endregion
        # region setup game screen
        self.field_x = width
        self.field_y = height
        self.screen_width = width * self.square_size
        self.screen_height = height * self.square_size + self.info_panel_height
        self.screen = pg.display.set_mode(size=[self.screen_width, self.screen_height])     # setup game screen
        # endregion
        # look through map array and generate corresponding game elements
        for y in range(height):
            for x in range(width):
                if map_template[y][x] == "w":   # wall
                    self.MAP_WALLS[y][x] = 1
                    self.walls_list.append(
                        spr.CustomSprite(x * self.square_size + self.square_size / 2,
                                         y * self.square_size + self.square_size / 2, 'Assets/wall.png'))
                elif map_template[y][x] == "t":     # tree
                    self.MAP_TREES[y][x] = 1
                    self.trees_list.append(
                        ent.Resource(spr.CustomSprite(x * self.square_size + self.square_size / 2,
                                                      y * self.square_size + self.square_size / 2, 'Assets/tree.png'), 10, 1))
                elif map_template[y][x] == "s":     # steel
                    self.MAP_STEEL[y][x] = 1
                    self.steel_list.append(
                        ent.Resource(spr.CustomSprite(x * self.square_size + self.square_size / 2,
                                                      y * self.square_size + self.square_size / 2, 'Assets/steel.png'), 10,
                                     1))
                elif map_template[y][x] == "p0":    # first agent
                    self.MAP_PLAYER0[y][x] = 1
                    self.p0 = Player(self.square_size, self.square_size, self.p1, x, y, self.square_size)
                elif map_template[y][x] == "p1":    # second agent
                    self.MAP_PLAYER1[y][x] = 1
                    self.p1 = Player(self.square_size, self.square_size, self.p0, x, y, self.square_size)
                elif map_template[y][x] == "S":     # station
                    self.MAP_STATION[y][x] = 1
                    self.station_x = x
                    self.station_y = y
                    self.station = spr.CustomSprite(x * self.square_size + self.square_size / 2,
                                                    y * self.square_size + self.square_size / 2, 'Assets/station.png')
                elif map_template[y][x] == "T":     # train and rail under it
                    self.MAP_TRAIN[y][x] = 1
                    self.MAP_RAILS[y][x] = 3  # 1: bad rails 2: path rails 3: used rails 4: last rail in path
                    self.train = ent.Train(x * self.square_size + self.square_size / 2,
                                           y * self.square_size + self.square_size / 2, self.square_size,
                                           'Assets/train_right.png', 0)
                    self.train.x = x
                    self.train.y = y
                    self.rail = spr.CustomSprite(x * self.square_size + self.square_size / 2,
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
                elif map_template[y][x] == "r":  # rail, use only for path rails!
                    rail = spr.CustomSprite(x * self.square_size + self.square_size / 2,
                                            y * self.square_size + self.square_size / 2, 'Assets/rail.png')
                    self.rail_list.append(rail)
                    self.rails = pg.sprite.Group(self.rail_list)
                else:
                    continue
        self.steel_list_sprites = self.get_entities_sprites(self.steel_list)
        self.trees_list_sprites = self.get_entities_sprites(self.trees_list)
        return

    def draw_grid(self, w, h, size):
        """
        draws grid

        Parameters:
            w:
                screen width in pixels
            h:
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
        Gets entities sprites.

        Returns:
            array of sprites from entities
        Parameters:
            entities: array of entity type
        """
        sprt = []
        for i in entities:
            sprt.append(i.sprite)
        return sprt

    def draw_game_info(self, agent_rewards: []):
        """
        draws UI on the bottom of the game
        :parameter agent_rewards: array of the current agent rewards
        """
        msg0 = 'episode %d' % self.episode
        msg = 'trees: ' + str(self.collected_trees)
        msg += '   steel: ' + str(self.collected_steel)
        msg += '   rails: ' + str(self.collected_rails)
        if self.tick_count < self.train.start_delay:
            # how many ticks before train departs
            msg += '    train departs in: ' + str((self.train.delay - self.tick_count))
        msg1 = 'rewards:'
        for i in agent_rewards:
            msg1 += ' %.2f,' % i
        info_text = self.text_font.render(msg0, True, 'black')
        self.screen.blit(info_text, [10, self.screen.get_height() - self.info_panel_height * 95 / 100])
        info_text = self.text_font.render(msg, True, 'black')
        self.screen.blit(info_text, [10, self.screen.get_height() - self.info_panel_height * 65 / 100])
        info_text = self.text_font.render(msg1[:-1], True, 'black')
        self.screen.blit(info_text, [10, self.screen.get_height() - self.info_panel_height * 35 / 100])

    def env_action_update(self, action, player: Player, map_pl):
        """
        Is called every environment step(). Selected agent interacts with the game based on the action
        :param action: number selected by neural network
        :param player: which agent (player) is going now
        :param map_pl: game map of the current agent's position
        :return: collected reward
        """
        reward = 0
        if action is None or self.rail_path_completed:
            return reward
        PLACE_RAIL = False
        REMOVE_RAIL = False
        # region do stuff based on chosen action (actions are defined in the unrailed_env)
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
        # endregion
        player.set_dir()
        # check collisions in front of the agent
        collide_trees = player.collider.collidelist(self.trees_list_sprites)
        collide_steel = player.collider.collidelist(self.steel_list_sprites)
        collide_train = player.collider.colliderect(self.train.sprite)
        collide_station = player.collider.colliderect(self.station)
        collide_walls = player.collider.collidelist(self.walls_list)
        # if nothing 'solid' is in front of the agent, and it decided to move then change position
        if collide_trees == -1 and collide_steel == -1 and not collide_train \
                and not collide_station and collide_walls == -1:
            if 0 <= action < 4:
                map_pl[player.y][player.x] = 0
                player.set_pos()
                map_pl[player.y][player.x] = 1
        x = player.collider.x // self.square_size   # calculate grid x
        y = player.collider.y // self.square_size

        reward += REWARDS["every step"]
        if PLACE_RAIL:   # rails are placed UNDER the agent
            # if under the agent is empty and there are rails available then place rail under the agent
            if player.rect.collidelist(self.rail_list) == -1 and player.rect.collidelist(self.rail_path) == -1:
                if self.collected_rails >= 1 and player.rect.collidelist(self.used_rail_list) == -1:
                    # we need later to check whether agent placed bad rail
                    placed_bad_rails_len_before = len(self.rail_list)
                    self.collected_rails -= 1
                    self.MAP_RAILS[y][x] = 1
                    rail = spr.CustomSprite(player.rect.x + self.square_size / 2,
                                            player.rect.y + self.square_size / 2,
                                      'Assets/rail.png')
                    self.rail_list.append(rail)
                    self.rails = pg.sprite.Group(self.rail_list)

                    mid = self.calculate_rail_path()
                    reward += mid
                    # check after path was formed how many bad rails are on the field now
                    if len(self.rail_list) > placed_bad_rails_len_before:
                        reward += REWARDS["bad rail placed"]
        elif REMOVE_RAIL:   # rail is removed under the agent too
            # if there is bad rail under
            if player.rect.collidelist(self.rail_list) != -1:
                reward += REWARDS["bad rail removed"]
                self.collected_rails += 1
                self.rail_list.pop(player.rect.collidelist(self.rail_list))
                self.MAP_RAILS[y][x] = 0
                self.rails = pg.sprite.Group(self.rail_list)
            # if there is last path rail under
            elif len(self.rail_path) > 0 and player.rect.colliderect(self.rail_path[-1]):
                self.collected_rails += 1
                self.MAP_RAILS[y][x] = 0
                self.rail_path.pop(player.rect.collidelist(self.rail_path))
                self.rail_paths = pg.sprite.Group(self.rail_path)
                if len(self.rail_path) == 0:
                    x = self.used_rail_list[-1].rect.x
                    y = self.used_rail_list[-1].rect.y
                else:
                    x = self.rail_path[-1].rect.x
                    y = self.rail_path[-1].rect.y
                reward += REWARDS["path removed"]
                self.last_rail_colliders = [
                    pg.rect.Rect(x, y - self.square_size, self.square_size, self.square_size),  # up
                    pg.rect.Rect(x, y + self.square_size, self.square_size, self.square_size),  # down
                    pg.rect.Rect(x + self.square_size, y, self.square_size, self.square_size),  # right
                    pg.rect.Rect(x - self.square_size, y, self.square_size, self.square_size)]  # left
                # recalculate path again, because there might be bad rails nearby
                mid = self.calculate_rail_path()
                reward += mid
        elif collide_train:
            # craft rail
            if self.collected_trees >= self.trees_needed and self.collected_steel >= self.steel_needed:
                self.collected_trees -= self.trees_needed
                self.collected_steel -= self.steel_needed
                self.collected_rails += 1
                reward += REWARDS["made rail"]
        elif collide_trees != -1:
            # cut down tree
            reward += REWARDS["tree/steel collided"]
            if self.trees_list[collide_trees].damage():
                reward += REWARDS["got tree/steel"]
                if self.collected_trees < self.collected_steel:
                    reward += REWARDS["bonus"]
                self.collected_trees += self.trees_list[collide_trees].loot
                self.trees_list.pop(collide_trees)
                self.MAP_TREES[y][x] = 0
                self.trees_list_sprites = self.get_entities_sprites(self.trees_list)
                self.trees = pg.sprite.Group(self.trees_list_sprites)
        elif collide_steel != -1:
            # mine steel
            reward += REWARDS["tree/steel collided"]
            if self.steel_list[collide_steel].damage():
                reward += REWARDS["got tree/steel"]
                if self.collected_trees > self.collected_steel:
                    reward += REWARDS["bonus"]
                self.collected_steel += self.steel_list[collide_steel].loot
                self.steel_list.pop(collide_steel)
                self.MAP_STEEL[y][x] = 0
                self.steel_list_sprites = self.get_entities_sprites(self.steel_list)
                self.steel = pg.sprite.Group(self.steel_list_sprites)
        return reward
