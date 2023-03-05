# Local imports
from .ship import Ship
from .object import GameObject
from .missile import Missile

# Library imports
import gym
import numpy as np
import shapely
import cv2
import math


class SinglePlayerGame(gym.Env):
    WINDOW_WIDTH = 250
    WINDOW_HEIGHT = 250
    OBSERVATION_SHAPE = (80, 80)
    SPAWN_ENEMIES_INTERVAL = 100
    ENEMY_MARGIN = 50
    ENEMY_SPEED = 4
    NOOP = 0
    ACCELERATE = 1
    DECELERATE = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    FIRE_MISSILE = 5
    STEP_REWARD = 0
    KILL_REWARD = 10
    PENALTY = -10

    def __init__(self):
        # Define the observation space
        self.observation_space = gym.spaces.Discrete(15, )
        self.arena = shapely.geometry.polygon.Polygon(
            [(0, 0), (self.WINDOW_WIDTH, 0), (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), (0, self.WINDOW_HEIGHT)])

        # Define the action space
        self.action_space = gym.spaces.Discrete(6, )

        # Create a canvas to draw the game on
        self.canvas = np.full((self.WINDOW_WIDTH, self.WINDOW_HEIGHT, 3), 255, dtype=np.uint8)

        # Create the player, enemies and missiles
        self.player = Ship(self.ENEMY_MARGIN, self.WINDOW_HEIGHT // 2, 0, 0, player=True)
        self.player.missile_buffer = Ship.PLAYER_MISSILE_THRESHOLD
        self.enemies = []
        self.player_missiles = []
        self.to_remove = []

        # Set time to 0
        self.time = 0

        # Draw the game on the canvas
        self.draw_game_on_canvas()

    def draw_element_on_canvas(self, element: GameObject):
        if self.arena.covers(element.hit_box):
            (minx, miny, maxx, maxy) = element.hit_box.bounds
            (minx, miny, maxx, maxy) = (int(minx), int(miny), int(maxx), int(maxy))
            self.canvas[self.WINDOW_HEIGHT - maxy:self.WINDOW_HEIGHT - miny, minx:maxx] = \
                element.icon[
                element.padded_icon_height - (element.padded_icon_height - (maxy - miny)) // 2 - (maxy - miny):
                element.padded_icon_height - (element.padded_icon_height - (maxy - miny)) // 2,
                (element.padded_icon_width - (maxx - minx)) // 2:
                (element.padded_icon_width - (maxx - minx)) // 2 + (maxx - minx)
                ]

    def draw_game_on_canvas(self):
        self.canvas = np.full((self.WINDOW_WIDTH, self.WINDOW_HEIGHT, 3), 255, dtype=np.uint8)
        self.draw_element_on_canvas(self.player)
        for enemy in self.enemies:
            self.draw_element_on_canvas(enemy)
        for missile in self.player_missiles:
            self.draw_element_on_canvas(missile)

    def reset(self):
        # Create a canvas to draw the game on
        self.canvas = np.full((self.WINDOW_WIDTH, self.WINDOW_HEIGHT, 3), 255, dtype=np.uint8)

        # Create the player, enemies and missiles
        self.player = Ship(self.ENEMY_MARGIN, self.WINDOW_HEIGHT // 2, 0, 0, player=True)
        self.player.missile_buffer = Ship.PLAYER_MISSILE_THRESHOLD
        self.enemies = []
        self.player_missiles = []

        # Set time back to 0
        self.time = 0

        # Draw the game on the canvas
        self.draw_game_on_canvas()

        return self.get_observation()

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array'], 'Invalid mode, must be either "human" or "rgb_array"'
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)
        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def preprocess_frame(self):
        resized = cv2.resize(self.canvas, self.OBSERVATION_SHAPE)
        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return grayscale

    def get_observation(self):
        # Get info about the player
        player_x = self.player.x
        player_y = self.player.y
        player_angle = (self.player.angle % 360) - 180
        player_x_off = (round(np.cos(np.radians(self.player.angle)) * self.player.speed))
        player_y_off = (round(np.sin(np.radians(self.player.angle)) * self.player.speed))
        player_info = [player_x, player_y, player_angle, player_x_off, player_y_off]

        # Get the nearest enemy
        enemy_start_x = (self.WINDOW_WIDTH - self.ENEMY_MARGIN)
        enemy_start_top_y = self.ENEMY_MARGIN
        enemy_start_mid_y = (self.WINDOW_HEIGHT // 2)
        enemy_start_bot_y = (self.WINDOW_HEIGHT - self.ENEMY_MARGIN)
        enemy_start_spawned = 0

        enemy_info = []
        for idx, enemy in enumerate(self.enemies):
            if idx > 0:
                break
            enemy_info.append(enemy.x)
            enemy_info.append(enemy.y)
            enemy_x_off = round(np.cos(np.radians(enemy.angle)) * enemy.speed)
            enemy_y_off = round(np.sin(np.radians(enemy.angle)) * enemy.speed)
            enemy_info.append(enemy_x_off)
            enemy_info.append(enemy_y_off)
            enemy_info.append(1)

        order = []
        if self.time % self.SPAWN_ENEMIES_INTERVAL > 2 * (self.SPAWN_ENEMIES_INTERVAL // 3):
            order = [enemy_start_mid_y, enemy_start_bot_y, enemy_start_top_y]
        elif self.time % self.SPAWN_ENEMIES_INTERVAL > self.SPAWN_ENEMIES_INTERVAL // 3:
            order = [enemy_start_top_y, enemy_start_mid_y, enemy_start_bot_y]
        elif self.time % self.SPAWN_ENEMIES_INTERVAL >= 0:
            order = [enemy_start_bot_y, enemy_start_top_y, enemy_start_mid_y]

        idx = 0
        while len(enemy_info) < 5:
            enemy_info.append(enemy_start_x)
            enemy_info.append(order[idx])
            enemy_info.append(0)
            enemy_info.append(0)
            enemy_info.append(enemy_start_spawned)
            idx += 1

        # Get 1 player missile
        missile_info = []

        for idx, missile in enumerate(self.player_missiles):
            if idx > 0:
                break
            missile_info.append(missile.x / self.WINDOW_WIDTH)
            missile_info.append(missile.y / self.WINDOW_HEIGHT)
            missile_x_off = round(np.cos(np.radians(missile.angle)) * missile.speed)
            missile_y_off = round(np.sin(np.radians(missile.angle)) * missile.speed)
            missile_info.append(missile_x_off)
            missile_info.append(missile_y_off)
            missile_info.append(1)

        while len(missile_info) < 5:
            missile_info.append(player_x)
            missile_info.append(player_y)
            missile_x_off = (round(np.cos(np.radians(self.player.angle)) * self.player.speed))
            missile_y_off = (round(np.sin(np.radians(self.player.angle)) * self.player.speed))
            missile_info.append(missile_x_off)
            missile_info.append(missile_y_off)
            missile_info.append(0)

        observation = player_info + enemy_info + missile_info
        return observation

    def step(self, action):
        done = False

        # Remove any elements
        for element in self.to_remove:
            if element in self.enemies:
                self.enemies.remove(element)
                self.to_remove.remove(element)
            if element in self.player_missiles:
                self.player_missiles.remove(element)
                self.to_remove.remove(element)

        # Apply the chosen action
        assert self.action_space.contains(action), "Invalid Action"
        if action == self.TURN_LEFT:
            self.player.rotate_left()
        elif action == self.TURN_RIGHT:
            self.player.rotate_right()
        elif action == self.ACCELERATE:
            self.player.accelerate()
        elif action == self.DECELERATE:
            self.player.decelerate()
        elif action == self.FIRE_MISSILE and self.player.can_fire_missile():
            x_off = round(np.cos(np.radians(self.player.angle)) * (self.player.speed + 16))
            y_off = round(np.sin(np.radians(self.player.angle)) * (self.player.speed + 16))
            x_off *= 2
            y_off *= 2
            if len(self.player_missiles) < 1:
                self.player_missiles.append(Missile(self.player.x + x_off, self.player.y + y_off, self.player.angle, player=True))

        # Update the state of all elements in the game world and draw them on the canvas
        self.player.move()

        if self.time % self.SPAWN_ENEMIES_INTERVAL == 0:
            enemy = Ship(self.WINDOW_WIDTH - self.ENEMY_MARGIN, self.WINDOW_HEIGHT // 2, 180, self.ENEMY_SPEED, player=False)
            self.enemies.append(enemy)
        elif self.time % self.SPAWN_ENEMIES_INTERVAL == self.SPAWN_ENEMIES_INTERVAL // 3:
            enemy = Ship(self.WINDOW_WIDTH - self.ENEMY_MARGIN, self.ENEMY_MARGIN, 180, self.ENEMY_SPEED, player=False)
            self.enemies.append(enemy)
        elif self.time % self.SPAWN_ENEMIES_INTERVAL == 2 * (self.SPAWN_ENEMIES_INTERVAL // 3):
            enemy = Ship(self.WINDOW_WIDTH - self.ENEMY_MARGIN, self.WINDOW_HEIGHT - self.ENEMY_MARGIN, 180, self.ENEMY_SPEED, player=False)
            self.enemies.append(enemy)

        # Update the enemies
        for enemy in self.enemies:
            # Angle the enemy towards the player and move them
            enemy_angle = enemy.angle % 360
            angle_to_player = math.degrees(math.atan2(self.player.y - enemy.y, self.player.x - enemy.x))
            if angle_to_player < 0:
                angle_to_player = 360 + angle_to_player
            if int(enemy_angle) != int(angle_to_player):
                turn_left_start = angle_to_player - 180
                if turn_left_start >= 0:
                    if turn_left_start < enemy_angle < angle_to_player:
                        enemy.rotate(min(enemy.turning_angle, angle_to_player - enemy_angle))
                    else:
                        enemy.rotate(max(-enemy.turning_angle, angle_to_player - enemy_angle))
                else:
                    turn_left_start = 360 + turn_left_start
                    if turn_left_start < enemy_angle or enemy_angle < angle_to_player:
                        enemy.rotate(min(enemy.turning_angle, angle_to_player - enemy_angle))
                    else:
                        enemy.rotate(max(-enemy.turning_angle, angle_to_player - enemy_angle))
            enemy.move()

            # Check for collisions
            for other_enemy in self.enemies:
                if other_enemy != enemy and enemy.hit_box.intersects(other_enemy.hit_box):
                    self.enemies.remove(enemy)
                    self.enemies.remove(other_enemy)

            if not self.arena.covers(enemy.hit_box):
                self.enemies.remove(enemy)

            if self.player.hit_box.intersects(enemy.hit_box):
                self.player.decrease_hp()

        # Update the missiles
        enemies_killed_by_missiles = 0
        for missile in self.player_missiles:
            missile.move()

            if not self.arena.covers(missile.hit_box):
                self.player_missiles.remove(missile)

            if missile not in self.player_missiles:
                continue

            for enemy in self.enemies:
                if enemy.hit_box.intersects(missile.hit_box):
                    self.to_remove.append(enemy)
                    self.to_remove.append(missile)
                    if missile.player:
                        enemies_killed_by_missiles += 1

        self.draw_game_on_canvas()

        # Calculate the reward
        reward = self.STEP_REWARD + (enemies_killed_by_missiles * self.KILL_REWARD)

        if not self.arena.covers(self.player.hit_box) or self.player.hp <= 0:
            done = True
            reward += self.PENALTY

        self.time += 1
        return self.get_observation(), reward, done, {}
