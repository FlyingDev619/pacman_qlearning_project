"""Chaser and ambusher ghost behavior."""

import random

from config import GHOST_SPEED_RATIO


class Ghost:
    """A ghost that moves at 70% of Pac-Man's tile-step rate.

    `target_rate` is the probability of following its target. On the remaining
    moves it picks a random legal neighboring cell.
    """

    def __init__(self, start_position, name, color, target_rate):
        self.position = start_position
        self.name = name
        self.color = color
        self.target_rate = target_rate
        self.speed_progress = 0  # Integer accumulator: 70 points per Pac-Man step.

    def should_move_this_turn(self):
        self.speed_progress += int(GHOST_SPEED_RATIO * 100)
        if self.speed_progress >= 100:
            self.speed_progress -= 100
            return True
        return False

    def choose_target(self, pacman, maze):
        raise NotImplementedError

    def move(self, maze, pacman):
        """Advance the speed clock and, when due, move one tile."""
        if not self.should_move_this_turn():
            return False

        legal = maze.legal_neighbors(self.position)
        if not legal:
            return False

        target = self.choose_target(pacman, maze)
        if random.random() < self.target_rate:
            next_position = maze.next_step_towards(self.position, target)
            if next_position == self.position:
                next_position = random.choice(legal)
        else:
            next_position = random.choice(legal)

        self.position = next_position
        return True

    def draw(self, surface, tile_size, offset_x, offset_y):
        import pygame

        x, y = self.position
        left = offset_x + x * tile_size
        top = offset_y + y * tile_size
        center_x = left + tile_size // 2
        body_top = top + 6
        radius = tile_size // 2 - 5

        pygame.draw.circle(surface, self.color, (center_x, body_top + radius), radius)
        body_rect = pygame.Rect(center_x - radius, body_top + radius - 1,
                                radius * 2, radius + 4)
        pygame.draw.rect(surface, self.color, body_rect)

        # Three small white scallops make the bottom edge read as a ghost.
        for index in range(3):
            scallop_x = center_x - radius + 4 + index * (radius * 2 - 8) / 2
            pygame.draw.circle(surface, (12, 16, 31),
                               (int(scallop_x), body_top + radius * 2 + 1), 4)

        eye_y = body_top + radius - 2
        pygame.draw.circle(surface, (255, 255, 255), (center_x - 5, eye_y), 4)
        pygame.draw.circle(surface, (255, 255, 255), (center_x + 5, eye_y), 4)
        pygame.draw.circle(surface, (25, 35, 65), (center_x - 4, eye_y), 2)
        pygame.draw.circle(surface, (25, 35, 65), (center_x + 6, eye_y), 2)


class Chaser(Ghost):
    def __init__(self, start_position, target_rate=0.8):
        super().__init__(start_position, "Chaser", (245, 75, 95), target_rate)

    def choose_target(self, pacman, maze):
        return pacman.position


class Ambusher(Ghost):
    def __init__(self, start_position, target_rate=0.6):
        super().__init__(start_position, "Ambusher", (70, 205, 220), target_rate)

    def choose_target(self, pacman, maze):
        return pacman.predict_position(maze, steps=3)
