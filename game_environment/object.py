# Library imports
from shapely import affinity
import shapely.geometry
import numpy as np
import cv2


class GameObject(object):
    def __init__(self, icon_path: str, x: int, y: int, angle: int, turning_angle: int, speed: int, min_speed: int, max_speed: int, hp: int,
                 hit_box: shapely.geometry.polygon.Polygon = None):
        # Initialise the objects variables
        if hit_box is None:
            hit_box = shapely.geometry.polygon.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        self.x = x
        self.y = y
        self.angle = angle
        self.turning_angle = turning_angle
        self.speed = speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.hp = hp
        self.hit_box = hit_box

        # Read in the icon
        self.icon = cv2.imread(icon_path)
        self.icon_width, self.icon_height, self.icon_channels = self.icon.shape
        self.background_colour = self.icon[0, 0]  # Make the background colour the top left pixel colour

        # Create a padded icon so that rotations don't clip
        self.padded_icon_width = self.icon_width * 2
        self.padded_icon_height = self.icon_height * 2
        self.padded_icon = np.full((self.padded_icon_width, self.padded_icon_height, 3), self.background_colour, dtype=np.uint8)
        width_margin = (self.padded_icon_width - self.icon_width) // 2
        height_margin = (self.padded_icon_height - self.icon_height) // 2
        self.padded_icon[height_margin:height_margin + self.icon_height, width_margin: width_margin + self.icon_width] = self.icon
        self.icon = self.padded_icon

    def set_hit_box(self, points: np.array = None):
        self.hit_box = shapely.geometry.polygon.Polygon(points)

    def move(self):
        x_off = round(np.cos(np.radians(self.angle)) * self.speed)
        y_off = round(np.sin(np.radians(self.angle)) * self.speed)
        self.x += x_off
        self.y += y_off
        self.hit_box = shapely.affinity.translate(self.hit_box, xoff=x_off, yoff=y_off)

    def rotate(self, degree):
        self.angle += degree
        self.hit_box = shapely.affinity.rotate(self.hit_box, degree, 'center')
        m = cv2.getRotationMatrix2D((self.padded_icon_width // 2, self.padded_icon_height // 2), self.angle, 1)
        self.icon = cv2.warpAffine(self.padded_icon, m, (self.padded_icon_width, self.padded_icon_height))

    def rotate_left(self):
        self.rotate(self.turning_angle)

    def rotate_right(self):
        self.rotate(-self.turning_angle)

    def accelerate(self):
        self.speed = min(self.speed + 1, self.max_speed)

    def decelerate(self):
        self.speed = max(self.speed - 1, self.min_speed)

    def decrease_hp(self):
        self.hp -= 1
