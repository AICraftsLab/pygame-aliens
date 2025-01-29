#!/usr/bin/env python
""" pygame.examples.aliens

Shows a mini game where you have to defend against aliens.

What does it show you about pygame?

* pg.sprite, the difference between Sprite and Group.
* dirty rectangle optimization for processing for speed.
* music with pg.mixer.music, including fadeout
* sound effects with pg.Sound
* event processing, keyboard handling, QUIT handling.
* a main loop frame limited with a game clock from pg.time.Clock
* fullscreen switching.


Controls
--------

* Left and right arrows to move.
* Space bar to shoot
* f key to toggle between fullscreen.

"""

import os
import random
from typing import List
import math
import sys

# import basic pygame modules
import pygame as pg

# see if we can load more than standard BMP
if not pg.image.get_extended():
    raise SystemExit("Sorry, extended image module required")


# game constants
MAX_SHOTS = 2  # most player bullets onscreen
ALIEN_ODDS = 22  # chances a new alien appears
MAX_ALIENS = 3
BOMB_ODDS = 60  # chances a new bomb will drop
ALIEN_RELOAD = 12  # frames between new aliens
#SCREENRECT = pg.Rect(0, 0, 720, 1472)  #640, 480
SCREENRECT = None

main_dir = os.path.split(os.path.abspath(__file__))[0]


def load_image(file):
    """loads an image, prepares it for play"""
    file = os.path.join(main_dir, "data", file)
    try:
        surface = pg.image.load(file)
    except pg.error:
        raise SystemExit(f'Could not load image "{file}" {pg.get_error()}')
    return surface.convert()


def load_sound(file):
    """because pygame can be compiled without mixer."""
    if not pg.mixer:
        return None
    file = os.path.join(main_dir, "data", file)
    try:
        sound = pg.mixer.Sound(file)
        return sound
    except pg.error:
        print(f"Warning, unable to load, {file}")
    return None


# Each type of game object gets an init and an update function.
# The update function is called once per frame, and it is when each object should
# change its current position and state.
#
# The Player object actually gets a "move" function instead of update,
# since it is passed extra information about the keyboard.


class Player(pg.sprite.Sprite):
    """Representing the player as a moon buggy type car."""

    speed = 10
    bounce = 24
    gun_offset = -11
    images: List[pg.Surface] = []
    nearest_n_aliens = MAX_ALIENS
    nearest_n_bombs = 3

    def __init__(self, shots_grp, all_grp, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=SCREENRECT.midbottom)
        self.reloading_time = 10
        self.reloading = 0
        self.origtop = self.rect.top
        self.facing = -1
        self.shots_grp = shots_grp
        self.all_grp = all_grp

    def move(self, direction):
        if direction:
            self.facing = direction
        self.rect.move_ip(direction * self.speed, 0)
        self.rect = self.rect.clamp(SCREENRECT)
        if direction < 0:
            self.image = self.images[0]
        elif direction > 0:
            self.image = self.images[1]
        self.rect.top = self.origtop - (self.rect.left // self.bounce % 2)
    
    def get_direction(self):
        return 0 if self.facing < 0 else 1
    
    def shoot(self):
        if self.reloading <= 0 and len(self.shots_grp) < MAX_SHOTS:
            Shot(self.gunpos(), self.shots_grp, self.all_grp)
            self.reloading = self.reloading_time
            #if pg.mixer and shoot_sound is not None:
            #    shoot_sound.play()
        
    def update(self, **kwargs):
        action = None
        
        for item, value in kwargs.items():
            if item == 'action':
                action = value
        
        if action == 'left':
            self.move(-1)
        elif action == 'right':
            self.move(1)
        elif action == 'shoot':
            self.shoot()
        elif action == 'idle':
            pass
            
        self.reloading -= 1
            
    
    def gunpos(self):
        pos = self.facing * self.gun_offset + self.rect.centerx
        return pos, self.rect.top
        
    def distace_to(self, sprite):
        return math.dist(self.rect.midtop, sprite.get_rect().midbottom)
    
    def get_nearest_n_pos(self, n, group, direction=False):
        positions = []
        for i, sprite in enumerate(group.sprites()):
            if i >= n:
                break
            if direction:
                dir = sprite.get_direction()
                x, y = sprite.get_position()
                positions.append((x, y, dir))
            else:
                positions.append(sprite.get_position())
            
        return positions
        
    def get_nearest_aliens_pos_and_dir(self, aliens_grp):
        return self.get_nearest_n_pos(self.nearest_n_aliens, aliens_grp, True)

    def get_nearest_bombs_pos(self, bombs_grp):
        return self.get_nearest_n_pos(self.nearest_n_bombs, bombs_grp)

    def get_data(self, aliens_grp, bombs_grp):
        n_shots = len(self.shots_grp)
        shots_positions = []
        
        for shot in self.shots_grp:
            shots_positions.append(shot.get_position())
            
        shots_remaining = MAX_SHOTS - n_shots
        
        for _ in range(shots_remaining):
            shots_positions.append((0, 0))
        
        nearest_aliens = self.get_nearest_aliens_pos_and_dir(aliens_grp)
        nearest_bombs = self.get_nearest_bombs_pos(bombs_grp)
        
        aliens_remaining = self.nearest_n_aliens - len(nearest_aliens)
        bombs_remaining = self.nearest_n_bombs - len(nearest_bombs)
        
        for _ in range(aliens_remaining):
            nearest_aliens.append((0, 0, 0))
            
        for _ in range(bombs_remaining):
            nearest_bombs.append((0, 0))
            
        data = []
        #data_heading = []
        
        data.extend(self.rect.midtop)  # 2
        data.append(self.get_direction())  # 1
        data.append(int(self.reloading > 0))  # 1
        data.append(int(n_shots >= MAX_SHOTS))  # 1
        
        #data_heading.extend(['selfx, selfy'])
        #data_heading.append('direction')
        #data_heading.append('is_reloading')
        #data_heading.append('MaxShots')
        
        for pos in shots_positions:  # MAX_SHOTS 2x2=4
            data.extend(pos)
            #data_heading.extend(['shotx', 'shoty'])
        
        for pos_dir in nearest_aliens:  # self.nearest_aliens 3x3=9
            data.extend(pos_dir)
            #data_heading.extend(['alienx', 'alieny', 'aliendir'])
            
        for pos in nearest_bombs:  # self.nearest_bombs 3x2=6
            data.extend(pos)
            #data_heading.extend(['bombx', 'bomby'])
        
        return data
        
class Alien(pg.sprite.Sprite):
    """An alien space ship. That slowly moves down the screen."""

    speed = 13
    animcycle = 12
    images: List[pg.Surface] = []

    def __init__(self, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.facing = random.choice((-1, 1)) * Alien.speed
        self.frame = 0
        if self.facing < 0:
            self.rect.right = SCREENRECT.right

    def update(self, *args, **kwargs):
        self.rect.move_ip(self.facing, 0)
        if not SCREENRECT.contains(self.rect):
            self.facing = -self.facing
            self.rect.top = self.rect.bottom + 1
            self.rect = self.rect.clamp(SCREENRECT)
        self.frame = self.frame + 1
        self.image = self.images[self.frame // self.animcycle % 3]

    def get_position(self):
        return self.rect.midbottom
        
    def get_direction(self):
        return 0 if self.facing < 0 else 1
        
class Explosion(pg.sprite.Sprite):
    """An explosion. Hopefully the Alien and not the player!"""

    defaultlife = 12
    animcycle = 3
    images: List[pg.Surface] = []

    def __init__(self, actor, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect(center=actor.rect.center)
        self.life = self.defaultlife

    def update(self, *args, **kwargs):
        """called every time around the game loop.

        Show the explosion surface for 'defaultlife'.
        Every game tick(update), we decrease the 'life'.

        Also we animate the explosion.
        """
        self.life = self.life - 1
        self.image = self.images[self.life // self.animcycle % 2]
        if self.life <= 0:
            self.kill()


class Shot(pg.sprite.Sprite):
    """a bullet the Player sprite fires."""

    speed = -11
    images: List[pg.Surface] = []

    def __init__(self, pos, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=pos)

    def update(self, *args, **kwargs):
        """called every time around the game loop.

        Every tick we move the shot upwards.
        """
        self.rect.move_ip(0, self.speed)
        if self.rect.top <= 0:
            self.kill()
            
    def get_position(self):
        return self.rect.midtop


class Bomb(pg.sprite.Sprite):
    """A bomb the aliens drop."""

    speed = 9
    images: List[pg.Surface] = []

    def __init__(self, alien, explosion_group, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=alien.rect.move(0, 5).midbottom)
        self.explosion_group = explosion_group

    def update(self, *args, **kwargs):
        """called every time around the game loop.

        Every frame we move the sprite 'rect' down.
        When it reaches the bottom we:

        - make an explosion.
        - remove the Bomb.
        """
        self.rect.move_ip(0, self.speed)
        if self.rect.bottom >= SCREENRECT.height - 10:
            Explosion(self, self.explosion_group)
            self.kill()
            
    def get_position(self):
        return self.rect.midbottom


class Episode(pg.sprite.Sprite):
    """to keep track of the episode."""

    def __init__(self, episode, position, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.font = pg.font.Font(None, 50)
        self.font.set_italic(1)
        self.color = "white"
        self.lastepisode = -1
        self.update(episode=episode)
        x, y = position
        y -= self.font.size('Episode:')[1]
        self.rect = self.image.get_rect().move((x, y))

    def update(self, *args, **kwargs):
        episode = kwargs.get('episode')
        
        if episode is None:
            episode = 'error'
            
        "We only update the episode in update() when it has changed."
        if episode != self.lastepisode:
            self.lastepisode = episode
            msg = f"Episode: {episode}"
            self.image = self.font.render(msg, 0, self.color)


class Score(pg.sprite.Sprite):
    """to keep track of the score."""

    def __init__(self, position, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.font = pg.font.Font(None, 30)
        self.font.set_italic(1)
        self.color = "white"
        self.lastscore = -1
        self.update(score=0)
        x, y = position
        y -= self.font.size('Score:')[1]
        self.rect = self.image.get_rect().move((x, y))

    def update(self, *args, **kwargs):
        score = kwargs.get('score')
        
        if score is None:
            score = 'error'
            
        "We only update the score in update() when it has changed."
        if score != self.lastscore:
            self.lastscore = score
            msg = f"Score: {score}"
            self.image = self.font.render(msg, 0, self.color)


class AliensEnv:
    def __init__(self, surface, episode=0, render=True, fps=40, play_sounds=False):
        self.surface = surface
        self.surfrect = surface.get_rect()
        global SCREENRECT
        SCREENRECT = self.surfrect
        self.width = self.surfrect.w
        self.height = self.surfrect.h
        self.render = render
        self.play_sounds = play_sounds
        self.n_actions = 4
        self.n_observations = 24
        self.fps = fps
        self.episode = episode
        
        # Initialize pygame
        if self.play_sounds:
            if pg.get_sdl_version()[0] == 2:
                pg.mixer.pre_init(44100, 32, 2, 1024)
        
        pg.init()
        
        if self.play_sounds:
            if pg.mixer and not pg.mixer.get_init():
                print("Warning, no sound")
                pg.mixer = None

        self.fullscreen = False
        # Set the display mode
        self.winstyle = 0  # |FULLSCREEN
        self.bestdepth = pg.display.mode_ok(self.surfrect.size, self.winstyle, 32)
        #screen = pg.display.set_mode(SCREENRECT.size, winstyle, bestdepth)
        if self.fullscreen:
            self.surface = pg.display.set_mode(self.surfrect.size, pg.FULLSCREEN, self.bestdepth)

        # Load images, assign to sprite classes
        # (do this before the classes are used, after screen setup)
        img = load_image("player1.gif")
        Player.images = [img, pg.transform.flip(img, 1, 0)]
        img = load_image("explosion1.gif")
        Explosion.images = [img, pg.transform.flip(img, 1, 1)]
        Alien.images = [load_image(im) for im in ("alien1.gif", "alien2.gif", "alien3.gif")]
        Bomb.images = [load_image("bomb.gif")]
        Shot.images = [load_image("shot.gif")]

        # decorate the game window
        icon = pg.transform.scale(Alien.images[0], (32, 32))
        pg.display.set_icon(icon)
        pg.display.set_caption("Pygame Aliens")
        pg.mouse.set_visible(1)

        # create the background, tile the bgd image
        bgdtile = load_image("background.gif")
        bgdtile = pg.transform.scale(bgdtile, (bgdtile.get_rect().width, self.surfrect.height))
        self.background = pg.Surface(self.surfrect.size)
        for x in range(0, self.width, bgdtile.get_width()):
            self.background.blit(bgdtile, (x, 0))

        # load the sound effects
        if self.play_sounds:
            self.boom_sound = load_sound("boom.wav")
            self.shoot_sound = load_sound("car_door.wav")
            if pg.mixer:
                self.music = os.path.join(main_dir, "data", "house_lo.wav")
                pg.mixer.music.load(self.music)
                pg.mixer.music.play(-1)

        self.clock = pg.time.Clock()
        self.clicked = False
    
    def reset(self):
        self.score = 0
        
        # Initialize Game Groups
        self.aliens = pg.sprite.Group()
        self.shots = pg.sprite.Group()
        self.bombs = pg.sprite.Group()
        self.texts = pg.sprite.Group()
        self.all = pg.sprite.RenderUpdates()
        self.lastalien = pg.sprite.GroupSingle()
        self.alienreload = ALIEN_RELOAD
        self.episode += 1
        
        # initialize our starting sprites
        self.player = Player(self.shots, self.all, self.all)
        Alien(self.aliens, self.all, self.lastalien)
        if pg.font:
            text_pos = (50, self.height)
            episode = Episode(self.episode, text_pos, self.texts, self.all)
            #self.all.add(episode)
            
            text_pos = (50, episode.rect.top)
            Score(text_pos, self.texts, self.all)
            
        if self.render:
            self.render_()
        else:
            self.render_texts()
        
        observation = self.player.get_data(bombs_grp=self.bombs, aliens_grp=self.aliens)
        info = self.get_info()
        
        return observation, info
        
    def get_info(self):
        return f'Episode:{self.episode} Kills:{self.score}'
    
    def render_texts(self):
        self.surface.fill('black')
        self.texts.draw(self.surface)
        pg.display.flip()
    
    def render_(self, pos=None, flip=False):
        # clear/erase the last drawn sprites
        self.surface.blit(self.background, (0, 0))
        
        #if flip or pos is not None:
        #    self.surface.blit(self.background, (0, 0))
        #else:
        #    self.all.clear(self.surface, self.background)
        
        if pos is not None:
            for i in range(4, len(pos), 2):  # positions start from index 4
                pos_ = (pos[i], pos[i+1])
                if pos_ != (0, 0):
                    self.draw_line(pos_)
                
        # draw the scene
        #dirty = self.all.draw(self.surface)
        #pg.display.update(dirty)
        
        self.all.draw(self.surface)
        pg.display.flip()    
        
        # cap the framerate at 40fps. Also called 40HZ or 40 times per second.
        self.clock.tick(self.fps)
    
    def step(self, action):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                sys.exit()
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.close()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_f:
                    if not self.fullscreen:
                        print("Changing to FULLSCREEN")
                        surface_backup = self.surface.copy()
                        self.surface = pg.display.set_mode(self.surfrect.size, self.winstyle | pg.FULLSCREEN, self.bestdepth)
                        self.surface.blit(surface_backup, (0, 0))
                    else:
                        print("Changing to windowed mode")
                        surface_backup = self.surface.copy()
                        self.surface = pg.display.set_mode(self.surfrect.size, self.winstyle, self.bestdepth)
                        self.surface.blit(surface_backup, (0, 0))
                        
                    pg.display.flip()
                    self.fullscreen = not self.fullscreen
            elif event.type == pg.MOUSEBUTTONDOWN:
                self.clicked = True
            elif event.type == pg.MOUSEBUTTONUP:
                self.clicked = False
        
        step_reward = 0
            
        # check for episode termination 
        termination = not self.player.alive() or self.score >= 100

        # update all the sprites
        if action == 0:
            action = 'right'
        elif action == 1:
            action = 'left'
        elif action == 2:
            action = 'shoot'
        elif action == 3:
            action = 'idle'
            
        self.all.update(action=action, score=self.score, episode=self.episode)
        
        # Create new alien
        if self.alienreload >= 0:
            self.alienreload -= 1
        elif len(self.aliens) < MAX_ALIENS and not int(random.random() * ALIEN_ODDS):
            Alien(self.aliens, self.all, self.lastalien)
            self.alienreload = ALIEN_RELOAD

        # Drop bombs
        if self.lastalien and not int(random.random() * BOMB_ODDS):
            Bomb(self.lastalien.sprite, self.all, self.bombs, self.all)

        # Detect collisions between aliens and players.
        for alien in pg.sprite.spritecollide(self.player, self.aliens, 1):
            if self.play_sounds and pg.mixer and self.boom_sound is not None:
                self.boom_sound.play()
            Explosion(alien, self.all)
            Explosion(self.player, self.all)
            step_reward -= 3
            self.player.kill()

        # See if shots hit the aliens.
        for alien in pg.sprite.groupcollide(self.aliens, self.shots, 1, 1).keys():
            if self.play_sounds and pg.mixer and self.boom_sound is not None:
                self.boom_sound.play()
            Explosion(alien, self.all)
            step_reward += 1
            self.score += 1

        # See if alien bombs hit the player.
        for bomb in pg.sprite.spritecollide(self.player, self.bombs, 1):
            if self.play_sounds and pg.mixer and self.boom_sound is not None:
                self.boom_sound.play()
            Explosion(self.player, self.all)
            Explosion(bomb, self.all)
            step_reward -= 2
            self.player.kill()
            
        observation = self.player.get_data(bombs_grp=self.bombs, aliens_grp=self.aliens)
        info = self.get_info()
        
        if self.render:
            self.render_()
        else:
            if self.clicked:
                #self.render_(pos=observation)
                self.render_()
            else:
                self.render_texts()
            
        return observation, step_reward, termination, info
    
    def draw_line(self, pos):
        pg.draw.line(self.surface, 'red', self.player.rect.midtop, pos)
    
    def close(self):
        if self.play_sounds:
            if pg.mixer:
                pg.mixer.music.fadeout(1000)
            pg.time.wait(1000)
        pg.quit()

# call the "main" function if running this script
if __name__ == "__main__":
    surface = pg.display.set_mode((840, 660))
    env = AliensEnv(episode=-1, surface=surface, render=True, play_sounds=False)
    
    for i in range(200):
        done = False
        observation, info = env.reset()
        episode_reward = 0
        
        while not done:
            action = random.randrange(env.n_actions)
            observation_, reward, done, info = env.step(action)
            episode_reward += reward
            #print(reward)
        print(info, 'Reward:', episode_reward)
        
    env.close()
