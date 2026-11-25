"""Pac-Man entity and its simple forward prediction for the ambusher."""

from config import ACTION_VECTORS


class Pacman:
    def __init__(self, start_position):
        self.position = start_position
        self.direction = "RIGHT"

    def move(self, action, maze):
        """Move one tile when legal. Return True if Pac-Man moved."""
        dx, dy = ACTION_VECTORS[action]
        destination = (self.position[0] + dx, self.position[1] + dy)
        if not maze.is_walkable(destination):
            return False
        self.position = destination
        self.direction = action
        return True

    def predict_position(self, maze, steps=3):
        """Predict a point up to `steps` tiles ahead using the current heading.

        At a junction, the prediction keeps moving straight. If blocked, it
        takes the first available turn in left/right/reverse order. The target
        is a simple estimate, not a claim about the Q-learning agent's future.
        """
        direction_order = {
            "UP": ("UP", "LEFT", "RIGHT", "DOWN"),
            "RIGHT": ("RIGHT", "UP", "DOWN", "LEFT"),
            "DOWN": ("DOWN", "RIGHT", "LEFT", "UP"),
            "LEFT": ("LEFT", "DOWN", "UP", "RIGHT"),
        }
        position = self.position
        heading = self.direction
        for _ in range(steps):
            next_position = None
            for action in direction_order[heading]:
                dx, dy = ACTION_VECTORS[action]
                candidate = (position[0] + dx, position[1] + dy)
                if maze.is_walkable(candidate):
                    next_position = candidate
                    heading = action
                    break
            if next_position is None:
                break
            position = next_position
        return position

    def draw(self, surface, tile_size, offset_x, offset_y):
        """Draw a Pac-Man face with an open, forward-facing mouth."""
        import math
        import pygame

        x, y = self.position
        center = (offset_x + x * tile_size + tile_size // 2,
                  offset_y + y * tile_size + tile_size // 2)
        radius = tile_size // 2 - 4
        pygame.draw.circle(surface, (255, 213, 56), center, radius)

        # Cut a small triangular wedge out of the circle to make the eating mouth.
        angle_by_direction = {"RIGHT": 0, "DOWN": 90, "LEFT": 180, "UP": 270}
        angle = math.radians(angle_by_direction[self.direction])
        spread = math.radians(31)
        mouth_points = [
            center,
            (center[0] + math.cos(angle - spread) * radius,
             center[1] + math.sin(angle - spread) * radius),
            (center[0] + math.cos(angle + spread) * radius,
             center[1] + math.sin(angle + spread) * radius),
        ]
        pygame.draw.polygon(surface, (12, 16, 31), mouth_points)
