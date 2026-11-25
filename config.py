"""Shared settings for the Pac-Man Q-learning project."""

GRID_SIZE = 15
TILE_SIZE = 36
HUD_TOP = 88
WINDOW_WIDTH = 1160
WINDOW_HEIGHT = 740

# Pac-Man performs one tile move this often. Each ghost moves 70% as often.
PACMAN_STEP_SECONDS = 0.14
GHOST_SPEED_RATIO = 0.70

MAX_STEPS_PER_EPISODE = 900
EPISODE_BREAK_SECONDS = 0

# Actions are kept in a fixed order throughout the project.
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
ACTION_VECTORS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}

PACMAN_START = (1, 1)
CHASER_START = (13, 1)
AMBUSHER_START = (1, 13)

# Ghosts use their target-seeking behavior with these probabilities.
CHASER_RATE = 0.8
AMBUSHER_RATE = 0.6

WINDOW_TITLE = "Pac-Man | Q-Learning"
