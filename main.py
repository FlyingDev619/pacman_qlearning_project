"""Pygame interface for the Pac-Man Q-learning project.

Run with: python main.py
Controls: SPACE pause/resume, UP/DOWN change training speed, R restart,
S save the Q-table, ESC quit. The speed has no upper limit.
"""

import argparse
import os

import pygame

from agent import QLearningAgent
from autosave import AutoSaver
from config import (
    HUD_TOP, PACMAN_STEP_SECONDS, TILE_SIZE,
    WINDOW_HEIGHT, WINDOW_TITLE, WINDOW_WIDTH,
)
from game import Game


# A restrained arcade palette: deep navy surfaces, electric-blue walls,
# warm pellets, and distinct colors for the two ghosts.
BG = (8, 12, 27)
HEADER = (13, 20, 40)
PANEL = (17, 26, 49)
PANEL_ALT = (22, 34, 61)
PANEL_EDGE = (40, 57, 91)
FLOOR = (12, 19, 37)
FLOOR_LINE = (19, 29, 51)
WALL = (35, 76, 151)
WALL_HI = (76, 132, 218)
PELLET = (255, 222, 150)
WHITE = (239, 245, 255)
MUTED = (139, 158, 194)
DIM = (83, 101, 137)
YELLOW = (255, 211, 65)
GREEN = (90, 220, 158)
RED = (247, 91, 111)
CYAN = (80, 207, 226)
AUTO_SAVE_INTERVAL = 100


class ArcadeUI:
    """Draw the game board and a compact live-training dashboard."""

    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts
        self.board_x = 32
        self.board_y = 132
        self.board_size = 15 * TILE_SIZE
        self.panel = pygame.Rect(610, 108, 520, 250)

    def text(self, message, position, font="small", color=WHITE, anchor="topleft"):
        image = self.fonts[font].render(str(message), True, color)
        rect = image.get_rect()
        setattr(rect, anchor, position)
        self.screen.blit(image, rect)
        return rect

    def rounded_panel(self, rect, color=PANEL, border=PANEL_EDGE, radius=14):
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)
        if border:
            pygame.draw.rect(self.screen, border, rect, width=1, border_radius=radius)

    def draw_background(self, game, speed, paused):
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, HEADER, (0, 0, WINDOW_WIDTH, HUD_TOP))
        pygame.draw.line(self.screen, PANEL_EDGE, (0, HUD_TOP - 1),
                         (WINDOW_WIDTH, HUD_TOP - 1), 1)

        # Brand mark and title.
        pygame.draw.circle(self.screen, YELLOW, (39, 41), 17)
        pygame.draw.polygon(self.screen, HEADER, [(39, 41), (56, 31), (56, 51)])
        self.text("PAC-MAN", (68, 18), "title", WHITE)
        self.text("Q-LEARNING ARCADE", (70, 54), "tiny", CYAN)

        # Session status at the right side of the header.
        status = "PAUSED" if paused else (
            "EPISODE COMPLETE" if game.terminal else "TRAINING LIVE"
        )
        status_color = YELLOW if paused or game.terminal else GREEN
        pill = pygame.Rect(WINDOW_WIDTH - 250, 27, 218, 34)
        pygame.draw.rect(self.screen, PANEL_ALT, pill, border_radius=17)
        pygame.draw.circle(self.screen, status_color, (pill.x + 17, pill.centery), 5)
        self.text(status, (pill.x + 31, pill.centery), "small", status_color, "midleft")
        self.text(f"{speed:.1f}x", (WINDOW_WIDTH - 270, pill.centery),
                  "small", MUTED, "midright")

    def draw_board_heading(self, game):
        self.text("LIVE MAZE", (self.board_x, 101), "medium", WHITE)
        self.text(
            f"15 × 15 GRID    •    {len(game.maze.floor)} PELLET CELLS",
            (self.board_x + 118, 105), "tiny", MUTED,
        )

    def draw_maze(self, game):
        tile = TILE_SIZE
        board_rect = pygame.Rect(self.board_x - 8, self.board_y - 8,
                                 self.board_size + 16, self.board_size + 16)
        self.rounded_panel(board_rect, PANEL, PANEL_EDGE, 14)

        for y in range(game.maze.height):
            for x in range(game.maze.width):
                rect = pygame.Rect(self.board_x + x * tile,
                                   self.board_y + y * tile, tile, tile)
                position = (x, y)
                if position in game.maze.walls:
                    pygame.draw.rect(self.screen, WALL, rect, border_radius=6)
                    inner = rect.inflate(-5, -5)
                    pygame.draw.rect(self.screen, WALL_HI, inner,
                                     width=1, border_radius=4)
                    # Small highlight gives the wall blocks a soft raised look.
                    pygame.draw.line(self.screen, (100, 157, 235),
                                     (rect.x + 8, rect.y + 5),
                                     (rect.right - 8, rect.y + 5), 1)
                else:
                    pygame.draw.rect(self.screen, FLOOR, rect)
                    pygame.draw.rect(self.screen, FLOOR_LINE, rect, width=1)
                    if position in game.maze.pellets:
                        center = rect.center
                        pygame.draw.circle(self.screen, PELLET, center, 3)
                        pygame.draw.circle(self.screen, (255, 247, 220),
                                           (center[0] - 1, center[1] - 1), 1)

        # A thin frame cleanly defines the board boundary.
        pygame.draw.rect(
            self.screen, PANEL_EDGE,
            pygame.Rect(self.board_x, self.board_y, self.board_size, self.board_size),
            width=1,
        )

        # Draw ghosts first so Pac-Man remains clearly visible at overlaps.
        for ghost in game.ghosts:
            ghost.draw(self.screen, tile, self.board_x, self.board_y)
        game.pacman.draw(self.screen, tile, self.board_x, self.board_y)

    def metric_card(self, x, y, width, label, value, accent):
        rect = pygame.Rect(x, y, width, 68)
        self.rounded_panel(rect, PANEL_ALT, None, 10)
        pygame.draw.rect(self.screen, accent,
                         pygame.Rect(x, y + 10, 3, 48), border_radius=2)
        self.text(label.upper(), (x + 13, y + 10), "tiny", MUTED)
        self.text(value, (x + 13, y + 32), "value", accent)

    def draw_dashboard(self, game, paused):
        """Show the requested metrics without clearance or progress sections."""
        x = self.panel.x + 18
        right = self.panel.right - 18
        content_width = right - x
        self.rounded_panel(self.panel, PANEL, PANEL_EDGE, 16)

        self.text("TRAINING OVERVIEW", (x, 127), "section", WHITE)
        self.text("Live reinforcement-learning metrics", (x, 151), "tiny", MUTED)

        gap = 9
        card_width = (content_width - 2 * gap) // 3
        row1 = 184
        self.metric_card(x, row1, card_width, "Episode", f"{game.episode:,}", YELLOW)
        self.metric_card(x + card_width + gap, row1, card_width,
                         "Learned states", f"{len(game.agent.q_table):,}", CYAN)
        self.metric_card(x + 2 * (card_width + gap), row1, card_width,
                         "Win rate", f"{game.win_rate:.1f}%", GREEN)

        # Display both counts as plain numbers, with no progress bar.
        row2 = 266
        wide_card = (content_width - gap) // 2
        self.metric_card(x, row2, wide_card, "Remaining pellets",
                         f"{len(game.maze.pellets):,}", YELLOW)
        self.metric_card(x + wide_card + gap, row2, wide_card,
                         "Episode steps", f"{game.step_count:,}", CYAN)


    def draw_footer(self):
        footer_y = WINDOW_HEIGHT - 27
        controls = [
            ("SPACE", "pause"), ("↑ / ↓", "speed"),
            ("R", "restart"), ("S", "save"), ("ESC", "quit"),
        ]
        x = 32
        for key, label in controls:
            key_surface = self.fonts["tiny"].render(key, True, WHITE)
            key_rect = key_surface.get_rect(topleft=(x, footer_y))
            pill = key_rect.inflate(12, 7)
            pygame.draw.rect(self.screen, PANEL_ALT, pill, border_radius=5)
            self.screen.blit(key_surface, key_rect)
            x = pill.right + 7
            self.text(label, (x, footer_y + 1), "tiny", MUTED)
            x += self.fonts["tiny"].size(label)[0] + 19

        self.text("TABULAR Q-LEARNING  •  15×15 MAZE",
                  (WINDOW_WIDTH - 32, footer_y + 1), "tiny", DIM, "topright")

    def draw(self, game, speed, paused):
        self.draw_background(game, speed, paused)
        self.draw_board_heading(game)
        self.draw_maze(game)
        self.draw_dashboard(game, paused)
        self.draw_footer()
        pygame.display.flip()


def parse_args():
    parser = argparse.ArgumentParser(description="Pac-Man with a tabular Q-learning agent")
    parser.add_argument("--qtable", default="q_table.json",
                        help="file used to save the learned Q-table (default: q_table.json)")
    parser.add_argument("--load", action="store_true",
                        help="legacy option; existing Q-tables load automatically")
    parser.add_argument("--seed", type=int, default=None,
                        help="optional random seed for repeatable agent exploration")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="initial training speed multiplier (default: 1.0; no upper cap)")
    return parser.parse_args()


def main():
    args = parse_args()
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("arial", 27, bold=True),
        "section": pygame.font.SysFont("arial", 18, bold=True),
        "value": pygame.font.SysFont("arial", 25, bold=True),
        "medium": pygame.font.SysFont("arial", 16, bold=True),
        "small": pygame.font.SysFont("arial", 14, bold=True),
        "tiny": pygame.font.SysFont("arial", 11),
    }

    agent = QLearningAgent(seed=args.seed)
    # Resume automatically whenever the selected Q-table already exists.
    # This keeps the learned policy and lets online Q-learning continue.
    if os.path.exists(args.qtable):
        agent.load(args.qtable)
        print(f"Loaded Q-table from {args.qtable}; continuing training.")
    elif args.load:
        print(f"No Q-table found at {args.qtable}; starting fresh.")
    else:
        print(f"No Q-table found at {args.qtable}; starting fresh.")

    game = Game(agent)
    ui = ArcadeUI(screen, fonts)
    speed = max(0.1, args.speed)
    paused = False
    move_accumulator = 0.0
    auto_saver = AutoSaver(AUTO_SAVE_INTERVAL)
    running = True

    while running:
        delta_seconds = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed += 0.5
                elif event.key == pygame.K_DOWN:
                    speed = max(0.1, speed - 0.5)
                elif event.key == pygame.K_r:
                    game.episode += 1
                    game.reset_episode()
                    move_accumulator = 0.0
                elif event.key == pygame.K_s:
                    agent.save(args.qtable)
                    auto_saver.mark_saved(game.completed_episodes)
                    print(f"Saved Q-table at completed episode "
                          f"{game.completed_episodes:,} to {args.qtable}")

        if not paused:
            if game.terminal:
                game.update(delta_seconds)
                move_accumulator = 0.0
            else:
                move_accumulator += delta_seconds * speed
                while move_accumulator >= PACMAN_STEP_SECONDS and not game.terminal:
                    game.step()
                    move_accumulator -= PACMAN_STEP_SECONDS
                    try:
                        if auto_saver.save_if_due(
                            agent, args.qtable, game.completed_episodes
                        ):
                            print(
                                "Auto-saved Q-table after completed episode "
                                f"{auto_saver.last_saved_episode:,}"
                            )
                    except OSError as error:
                        print(f"Auto-save failed: {error}")

        ui.draw(game, speed, paused)

    # Keep learning progress when the window closes.
    try:
        agent.save(args.qtable)
        print(f"Saved Q-table to {args.qtable}")
    except OSError as error:
        print(f"Could not save Q-table: {error}")
    pygame.quit()


if __name__ == "__main__":
    main()
