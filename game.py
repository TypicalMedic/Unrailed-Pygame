import random

import pygame as pg
import Sprites as spr
import Entities as ent


def draw_grid(w, h, size):
    for i in range(h // size):
        for j in range(w // size):
            pg.draw.rect(screen, color=square_color,
                         rect=[size * j, size * i, size, size], width=1)


def get_entities_sprites(entities: []):
    sprt = []
    for i in entities:
        sprt.append(i.sprite)
    return sprt


def draw_game_info():
    msg = 'trees: ' + str(collected_trees)
    msg += '   steel: ' + str(collected_steel)
    msg += '   rails: ' + str(collected_rails)
    msg1 = 'train speed: '
    if pg.time.get_ticks() < train.start_delay:
        msg += '    train departs in: ' + str((train.start_delay - pg.time.get_ticks()) // 1000)
        msg1 += '0'
    else:
        msg1 += str(train.speed)
    info_text = text_font.render(msg, True, 'black')
    screen.blit(info_text, [10, screen.get_height() - info_panel_height * 9 / 10])
    info_text = text_font.render(msg1, True, 'black')
    screen.blit(info_text, [10, screen.get_height() - info_panel_height * 6 / 10])


pg.init()

info_panel_height = 100
screen = pg.display.set_mode(size=[600, 500])
pl_dir_pics = [pg.image.load('player_left.png').convert_alpha(), pg.image.load('player_right.png').convert_alpha(),
               pg.image.load('player_up.png').convert_alpha(), pg.image.load('player_down.png').convert_alpha()]
bgclr = (255, 255, 255)
square_size = 20
square_color = 'grey'
game_over = False
pl_x = 30
pl_y = 30
tick_amount = 30
collected_trees = 0
collected_steel = 0
collected_rails = 0
trees_needed = 1
steel_needed = 1

pl = spr.player(pl_x, pl_y, 'player_up.png', square_size, square_size)
train = ent.train(50, 50, square_size, 'train.png')
station = spr.sprite(250, 50, 'station.png')

rail_list = []
rails = pg.sprite.Group(rail_list)
used_rail_list = []
rail = spr.sprite(50, 50, 'usedrail.png')
used_rail_list.append(rail)
used_rails = pg.sprite.Group(used_rail_list)

wall_amount = 50
walls_list = []
for i in range(wall_amount):
    walls_list.append(
        spr.sprite(random.randint(0, screen.get_width() / square_size) * square_size - square_size / 2,
                   random.randint(0, (
                               screen.get_height() - info_panel_height) / square_size) * square_size - square_size / 2,
                   'wall.png'))
walls = pg.sprite.Group(walls_list)

trees_amount = 50
trees_list = []
trees_list_sprites = []
for i in range(trees_amount):
    trees_list.append(
        ent.Resource(spr.sprite(random.randint(0, screen.get_width() / square_size) * square_size - square_size / 2,
                                random.randint(0, (
                                        screen.get_height() - info_panel_height) / square_size) * square_size - square_size / 2,
                                'tree.png'),
                     10, 1))
trees_list_sprites = get_entities_sprites(trees_list)
trees = pg.sprite.Group(trees_list_sprites)

steel_amount = 50
steel_list = []
steel_list_sprites = []
for i in range(steel_amount):
    steel_list.append(
        ent.Resource(spr.sprite(random.randint(0, screen.get_width() / square_size) * square_size - square_size / 2,
                                random.randint(0, (
                                        screen.get_height() - info_panel_height) / square_size) * square_size - square_size / 2,
                                'steel.png'),
                     10, 1))
steel_list_sprites = get_entities_sprites(steel_list)
steel = pg.sprite.Group(steel_list_sprites)

dsp = pg.display
clock = pg.time.Clock()
screen.fill(bgclr)
text_font = pg.font.SysFont("bahnschrift", 25)

draw_grid(screen.get_width(), (screen.get_height() - info_panel_height), square_size)

screen.blit(pl.image, pl.rect)
screen.blit(train.sprite.image, train.sprite.rect)
dsp.update()

while not game_over:
    if pg.time.get_ticks() > train.delay:
        train.move_train(tick_amount, rail_list, used_rail_list, station)
        if train.sprite.rect.collidelist(rail_list) == -1 and train.sprite.rect.collidelist(used_rail_list) == -1:
            print('game over')
            game_over = True
        if train.sprite.rect.collidelist(rail_list) != -1:
            train.speed += train.speed * 0.1
            ur = rail_list.pop(train.sprite.rect.collidelist(rail_list))
            ur.image = pg.image.load('usedrail.png').convert_alpha()
            used_rail_list.append(ur)
            used_rails = pg.sprite.Group(used_rail_list)
            rails = pg.sprite.Group(rail_list)
        if train.sprite.rect.colliderect(station):
            print('game won!')
            game_over = True
    collide_trees = pl.collider.collidelist(trees_list_sprites)
    collide_steel = pl.collider.collidelist(steel_list_sprites)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            game_over = True
        is_dir_key_pressed = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_LEFT:
                pl.pl_dir = pl.direction['left']
                is_dir_key_pressed = True
            elif event.key == pg.K_RIGHT:
                pl.pl_dir = pl.direction['right']
                is_dir_key_pressed = True
            elif event.key == pg.K_UP:
                pl.pl_dir = pl.direction['up']
                is_dir_key_pressed = True
            elif event.key == pg.K_DOWN:
                pl.pl_dir = pl.direction['down']
                is_dir_key_pressed = True
            pl.set_dir()
            collide_trees = pl.collider.collidelist(trees_list_sprites)
            collide_steel = pl.collider.collidelist(steel_list_sprites)
            collide_train = pl.collider.colliderect(train.sprite)
            collide_station = pl.collider.colliderect(station)
            collide_walls = pl.collider.collidelist(walls_list)
            if collide_trees == -1 and collide_steel == -1 and not collide_train \
                    and not collide_station and collide_walls == -1:
                if is_dir_key_pressed:
                    pl.set_pos()
            if event.key == pg.K_SPACE:
                if collide_train:
                    if collected_trees >= trees_needed and collected_steel >= steel_needed:
                        collected_trees -= trees_needed
                        collected_steel -= steel_needed
                        collected_rails += 1
                elif pl.collider.collidelist(rail_list) == -1:
                    if collected_rails >= 1 and collide_steel == -1 \
                            and collide_trees == -1 and collide_walls == -1 \
                            and pl.collider.collidelist(used_rail_list) == -1:
                        collected_rails -= 1
                        rail = spr.sprite(pl.collider.x + square_size / 2, pl.collider.y + square_size / 2, 'rail.png')
                        rail_list.append(rail)
                        rails = pg.sprite.Group(rail_list)
                else:
                    collected_rails += 1
                    rail_list.pop(pl.collider.collidelist(rail_list))
                    rails = pg.sprite.Group(rail_list)

            # чисто чтобы смотреть где коллайдер, потом удалить?
            pl.set_dir()
    if collide_trees != -1:
        if trees_list[collide_trees].damage():
            collected_trees += trees_list[collide_trees].loot
            trees_list.pop(collide_trees)
            trees_list_sprites = get_entities_sprites(trees_list)
            trees = pg.sprite.Group(trees_list_sprites)
    if collide_steel != -1:
        if steel_list[collide_steel].damage():
            collected_steel += steel_list[collide_steel].loot
            steel_list.pop(collide_steel)
            steel_list_sprites = get_entities_sprites(steel_list)
            steel = pg.sprite.Group(steel_list_sprites)

    screen.fill(bgclr)
    draw_grid(screen.get_width(), (screen.get_height() - info_panel_height), square_size)
    pg.draw.rect(screen, rect=pl.collider, color='green')
    trees.draw(screen)
    steel.draw(screen)
    rails.draw(screen)
    walls.draw(screen)
    used_rails.draw(screen)
    screen.blit(station.image, station.rect)
    screen.blit(pl.image, pl.rect)
    screen.blit(train.sprite.image, train.sprite.rect)
    draw_game_info()
    pg.display.update()
    clock.tick(tick_amount)

dsp.quit()
