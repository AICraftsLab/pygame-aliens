""" pygame.examples.aliens
Shows a mini game where you have to defend against aliens.
"""

import os
from typing import List
import math
import sys
import pygame as pg

import numpy as np
import gymnasium as gym
from gymnasium import spaces

pg.font.init()


# game constants
MAX_SHOTS = 2  # max player bullets onscreen
ALIEN_ODDS = 22  # chances a new alien appears
MAX_ALIENS = 5
BOMB_ODDS = 60  # chances a new bomb will drop
ALIEN_RELOAD = 12  # frames between new aliens
#SCREENRECT = pg.Rect(0, 0, 720, 1472)  #640, 480
SCREENRECT = None
RNG = None

main_dir = ''  # root

def load_image(file, render_mode):
    """loads an image, prepares it for play"""
    file = os.path.join(main_dir, "data", file)
    try:
        surface = pg.image.load(file)
    except pg.error:
        raise SystemExit(f'Could not load image "{file}" {pg.get_error()}')

    if render_mode == "human":
        return surface.convert()
    else:
        return surface


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
    nearest_n_bombs = 5
    shoot_sound = None

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
            if pg.mixer and self.shoot_sound is not None:
                self.shoot_sound.play()
        
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
        
        data.extend(self.rect.midtop)  # 2
        data.append(self.get_direction())  # 1
        data.append(int(self.reloading > 0))  # 1
        data.append(max(0, self.reloading))  # 1
        data.append(int(n_shots >= MAX_SHOTS))  # 1
        
        for pos in shots_positions:  # MAX_SHOTS * 2
            data.extend(pos)
        
        for pos_dir in nearest_aliens:  # MAX_ALIENS * 3
            data.extend(pos_dir)
            
        for pos in nearest_bombs:  # Player.nearest_n_bombs * 2
            data.extend(pos)
        
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
        self.facing = RNG.choice((-1, 1)) * Alien.speed
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
        self.font = pg.font.SysFont("comicsans", 30)
        self.font.set_italic(1)
        self.color = "white"
        self.lastepisode = -1
        self.update(episode=episode)
        self.rect = self.image.get_rect().move(position)

    def update(self, *args, **kwargs):
        episode = kwargs.get('episode')
        
        if episode is None:
            episode = ''
            
        "We only update the episode in update() when it has changed."
        if episode != self.lastepisode:
            self.lastepisode = episode
            msg = f"Episode: {episode}"
            self.image = self.font.render(msg, 1, self.color)


class Score(pg.sprite.Sprite):
    """to keep track of the score."""

    def __init__(self, position, *groups):
        pg.sprite.Sprite.__init__(self, *groups)
        self.font = pg.font.SysFont("comicsans", 20)
        self.font.set_italic(1)
        self.color = "white"
        self.lastscore = -1
        self.update(score=0)
        self.rect = self.image.get_rect().move(position)

    def update(self, *args, **kwargs):
        score = kwargs.get('score')
        
        if score is None:
            score = ''
            
        "We only update the score in update() when it has changed."
        if score != self.lastscore:
            self.lastscore = score
            msg = f"Score: {score}"
            self.image = self.font.render(msg, 1, self.color)


class AliensEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "fps": 40}
    def __init__(self, render_mode=None, play_sounds=False):
        self.width = 640
        self.height = 480
        self.play_sounds = play_sounds
        self.episode = 0
        self.clicked = False  # Is screen clicked?
        
        # observation space
        # for each alien: alienX, alienY, alienDirection
        # for each bomb and shot: X, Y
        # playerX, playerY, playerDirection, reloading_time, is_reloading, canShoot
        obs_num = MAX_ALIENS                * 3 + \
                  Player.nearest_n_bombs    * 2 + \
                  MAX_SHOTS                 * 2 + \
                  6
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_num,),
            dtype=np.float32
        )
        
        # right, left, shoot, nothing
        self.action_space = spaces.Discrete(4)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid Render Mode"
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.window = pg.display.set_mode((self.width, self.height))
        self.clock = None
        self.background = None
        self.boom_sound = None
        self.shoot_sound = None
        
        self.surface = pg.Surface((self.width, self.height))
        self.surfrect = self.surface.get_rect()
        global SCREENRECT
        SCREENRECT = self.surfrect
        
        # Load images, assign to sprite classes
        img = load_image("player1.gif", render_mode)
        Player.images = [img, pg.transform.flip(img, 1, 0)]
        
        img = load_image("explosion1.gif", render_mode)
        Explosion.images = [img, pg.transform.flip(img, 1, 1)]
        
        Alien.images = [load_image(im, render_mode) for im in ("alien1.gif", "alien2.gif", "alien3.gif")]
        Bomb.images = [load_image("bomb.gif", render_mode)]
        Shot.images = [load_image("shot.gif", render_mode)]
    
    def _get_info(self):
        return dict(Episode=self.episode, Kills=self.score)
        
    def _get_obs(self):
        obs = self.player.get_data(bombs_grp=self.bombs, aliens_grp=self.aliens)
        return np.array(obs, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        global RNG
        RNG = self.np_random
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
        
        # initialize the starting sprites
        self.player = Player(self.shots, self.all, self.all)
        Alien(self.aliens, self.all, self.lastalien)
        
        if self.render_mode == "human" and pg.font.get_init():
            if not pg.font.get_init():
                pg.font.init()
            text_pos = (10, 0)
            episode = Episode(self.episode, text_pos, self.texts, self.all)
            
            text_pos = (10, episode.rect.bottom)
            Score(text_pos, self.texts, self.all)
            
        if self.render_mode == "human":
            self._render_frame()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self, show_lines=False):
        if self.render_mode == "human" and self.clock is None:
            #self.window = pg.display.set_mode((self.width, self.height))
            self.clock = pg.time.Clock()
            
            # decorate the game window
            icon = pg.transform.scale(Alien.images[0], (32, 32))
            pg.display.set_icon(icon)
            pg.display.set_caption("Pygame Aliens")
            pg.mouse.set_visible(1)
            
            # create the background, tile the bgd image
            bgdtile = load_image("background.gif", self.render_mode)
            bgdtile = pg.transform.scale(bgdtile, (bgdtile.get_rect().width, self.surfrect.height))
            self.background = pg.Surface(self.surfrect.size)
            for x in range(0, self.width, bgdtile.get_width()):
                self.background.blit(bgdtile, (x, 0))

            # load the sound effects
            if self.play_sounds:
                if pg.get_sdl_version()[0] == 2:
                    pg.mixer.pre_init(44100, 32, 2, 1024)
                    
                pg.mixer.init()
                if pg.mixer and not pg.mixer.get_init():
                    print("Warning, no sound")
                    pg.mixer = None
                
                if pg.mixer:
                    music = os.path.join(main_dir, "data", "house_lo.wav")
                    pg.mixer.music.load(music)
                    pg.mixer.music.play(-1)
                    
                    self.boom_sound = load_sound("boom.wav")
                    self.shoot_sound = load_sound("car_door.wav")
                    
                    Player.shoot_sound = self.shoot_sound
        
        self.surface.blit(self.background, (0, 0))
        
        nearest_aliens = self.player.get_nearest_aliens_pos_and_dir(self.aliens)
        nearest_bombs = self.player.get_nearest_bombs_pos(self.bombs)
        
        if show_lines:
            for x, y, _ in nearest_aliens:
                self.draw_line((x, y))
            
            for x, y in nearest_bombs:
                self.draw_line((x, y))
            
            for shot in self.shots:
                pos = shot.get_position()
                self.draw_line(pos)
        
        self.all.draw(self.surface)
        
        if self.render_mode == "human":
            self.window.blit(self.surface, (0, 0))
            pg.event.pump()
            pg.display.flip()  
            self.clock.tick(self.metadata["fps"])
        else:  # rgb_array
            array = np.transpose(
                np.array(pg.surfarray.pixels3d(self.surface)), axes=(1, 0, 2)
            )
            return array
    
    def step(self, action):
        if self.render_mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.close()
                    sys.exit()
                if event.type == pg.MOUSEBUTTONDOWN:
                    self.clicked = not self.clicked
                    
        
        step_reward = 0

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
        elif len(self.aliens) < MAX_ALIENS and not int(RNG.random() * ALIEN_ODDS):
            Alien(self.aliens, self.all, self.lastalien)
            self.alienreload = ALIEN_RELOAD

        # Drop bombs
        if self.lastalien and not int(RNG.random() * BOMB_ODDS):
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
            
        observation = self._get_obs()
        terminated = not self.player.alive()
        truncated = self.score >= 100
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(self.clicked)
            
        return observation, step_reward, terminated, truncated, info
    
    def draw_line(self, pos):
        pg.draw.line(self.surface, 'red', self.player.rect.midtop, pos)
    
    def close(self):
        if self.play_sounds:
            if pg.mixer.get_init():
                pg.mixer.music.fadeout(1000)
            pg.time.wait(1000)
        pg.quit()
