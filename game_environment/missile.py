# Local imports
from .object import GameObject

# Library imports
import numpy as np


class Missile(GameObject):
    MISSILE_SPEED = 12
    HP = 1

    def __init__(self, x: int, y: int, angle: int, player: bool):
        icon_path = f'./game_environment/sprites/missile.png'
        super(Missile, self).__init__(icon_path, x, y, 0, 0, self.MISSILE_SPEED, self.MISSILE_SPEED, self.MISSILE_SPEED, self.HP)

        # Some custom stuff for the missile class
        self.player = player
        points = np.zeros((4, 2))
        points[0] = (self.x - (self.icon_width // 4), self.y - (self.icon_height // 8))
        points[1] = (self.x - (self.icon_width // 4), self.y + (self.icon_height // 8))
        points[2] = (self.x + (self.icon_width // 4), self.y + (self.icon_height // 8))
        points[3] = (self.x + (self.icon_width // 4), self.y - (self.icon_height // 8))
        super().set_hit_box(points)
        super().rotate(angle)
