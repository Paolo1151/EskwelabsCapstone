
import matplotlib.pyplot as plt

from .vector import Vector

import math

class StarSystem:
    def __init__(self, size):
        self.size = size
        self.bodies = []

        self.fig, self.ax = plt.subplots(
            1,
            1,
            subplot_kw = {'projection': '3d'},
            figsize = (self.size / 50, self.size / 50)
        )

        self.fig.tight_layout()

    def add_body(self, body):
        self.bodies.append(body)

    def update_all(self):
        for body in self.bodies:
            body.move()
            body.draw()

    def draw_all(self):
        self.ax.set_xlim((-self.size / 2, self.size / 2))
        self.ax.set_ylim((-self.size / 2, self.size / 2))
        self.ax.set_zlim((-self.size / 2, self.size / 2))
        plt.pause(0.001)
        self.ax.clear()

class StarSystemBody:
    min_display_size = 10
    display_log_base = 1.3

    def __init__(
        self,
        star_system,
        mass,
        position=(0,0,0),
        velocity=(0,0,0)
    ):
        self.star_system = star_system
        self.mass = mass
        self.position = position
        self.velocity = Vector(*velocity)
        self.display_size = max(
            math.log(self.mass, self.display_log_base),
            self.min_display_size
        ),
        self.color = 'black'
        self.star_system.add_body(self)

    def move(self):
        self.position = (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1],
            self.position[2] + self.velocity[2],
        )

    def draw(self):
        self.star_system.ax.plot(
            *self.position,
            marker='o',
            markersize=self.display_size,
            color=self.color
        )