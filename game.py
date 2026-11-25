"""Game rules and episode management. The Pygame window lives in main.py."""

import itertools

from config import (
    ACTIONS, ACTION_VECTORS, AMBUSHER_START, AMBUSHER_RATE,
    CHASER_START, CHASER_RATE, MAX_STEPS_PER_EPISODE, PACMAN_START,
)
from agent import QLearningAgent
from ghost import Ambusher, Chaser
from maze import Maze
from pacman import Pacman


class Game:
    def __init__(self, agent=None):
        self.maze = Maze()
        self.agent = agent or QLearningAgent()
        self.episode = 1
        self.completed_episodes = 0
        self.wins = 0
        self.episode_history = []  # True for a win, False for a loss.
        self.step_count = 0
        self.last_result = "Learning..."
        self.terminal = False
        self.terminal_timer = 0.0
        self._goal_cache = None
        self.reset_episode(initial=True)

    def reset_episode(self, initial=False):
        self._goal_cache = None
        self.maze.reset_pellets()
        self.pacman = Pacman(PACMAN_START)
        self.chaser = Chaser(CHASER_START, CHASER_RATE)
        self.ambusher = Ambusher(AMBUSHER_START, AMBUSHER_RATE)
        self.ghosts = [self.chaser, self.ambusher]
        self.maze.consume_pellet(PACMAN_START)
        self.step_count = 0
        self.terminal = False
        self.terminal_timer = 0.0
        if not initial:
            self.last_result = "Learning..."

    def legal_action_indices(self, position=None):
        position = position or self.pacman.position
        legal = []
        for index, action in enumerate(ACTIONS):
            dx, dy = ACTION_VECTORS[action]
            destination = (position[0] + dx, position[1] + dy)
            if self.maze.is_walkable(destination):
                legal.append(index)
        return legal

    def _ambusher_prediction_from(self, position, heading, steps=3):
        """Match Pac-Man's forward prediction without changing his live state."""
        direction_order = {
            "UP": ("UP", "LEFT", "RIGHT", "DOWN"),
            "RIGHT": ("RIGHT", "UP", "DOWN", "LEFT"),
            "DOWN": ("DOWN", "RIGHT", "LEFT", "UP"),
            "LEFT": ("LEFT", "DOWN", "UP", "RIGHT"),
        }
        for _ in range(steps):
            for action in direction_order[heading]:
                dx, dy = ACTION_VECTORS[action]
                candidate = (position[0] + dx, position[1] + dy)
                if self.maze.is_walkable(candidate):
                    position = candidate
                    heading = action
                    break
            else:
                break
        return position

    def _ghost_collision_probability(self, destination, action, ghost):
        """Probability this ghost lands on a candidate Pac-Man tile this turn."""
        if destination == ghost.position:
            return 1.0  # Collision happens before the ghost turn.
        if ghost.speed_progress + 70 < 100:
            return 0.0  # Ghost's unchanged 70%-speed clock says it will not move.

        if ghost.name == "Chaser":
            target = destination
        else:
            target = self._ambusher_prediction_from(destination, action, steps=3)

        legal = self.maze.legal_neighbors(ghost.position)
        if not legal:
            return 0.0
        follow_step = self.maze.next_step_towards(ghost.position, target)
        random_probability = (1.0 / len(legal)
                              if destination in legal else 0.0)
        if follow_step == ghost.position:
            # Ghost.move falls back to a random legal neighbor in this case.
            return random_probability
        return (ghost.target_rate * (follow_step == destination)
                + (1.0 - ghost.target_rate) * random_probability)

    def safe_action_indices(self, threshold=0.16):
        """Keep exploration away from moves with a high immediate capture risk."""
        legal = self.legal_action_indices()
        safe = []
        for index in legal:
            action = ACTIONS[index]
            dx, dy = ACTION_VECTORS[action]
            destination = (self.pacman.position[0] + dx,
                           self.pacman.position[1] + dy)
            risk = 0.0
            for ghost in self.ghosts:
                if destination == ghost.position:
                    risk = 1.0
                    break
                risk = max(risk, self._ghost_collision_probability(
                    destination, action, ghost
                ))
            if risk < threshold:
                safe.append(index)
        return safe or legal

    @staticmethod
    def _relative_sector(origin, other):
        dx = (other[0] > origin[0]) - (other[0] < origin[0])
        dy = (other[1] > origin[1]) - (other[1] < origin[1])
        return (dy + 1) * 3 + (dx + 1)

    @staticmethod
    def _region_id(position):
        """Map a cell into one of the maze's 25 coarse 3x3 regions."""
        x, y = position
        return (y // 3) * 5 + (x // 3)

    def _region_counts(self):
        counts = [0] * 25
        for position in self.maze.pellets:
            counts[self._region_id(position)] += 1
        return counts

    def _strategic_goal(self):
        """Choose and cache a pellet target for the current game snapshot."""
        if self._goal_cache is not None:
            return self._goal_cache
        if not self.maze.pellets:
            self._goal_cache = (0, self.pacman.position, 0)
            return self._goal_cache

        grouped = [[] for _ in range(25)]
        for cell in self.maze.pellets:
            grouped[self._region_id(cell)].append(cell)
        distances = self.maze._distances.get(self.pacman.position, {})
        nearest_ghost = min(
            self.maze.distance(self.pacman.position, ghost.position)
            for ghost in self.ghosts
        )
        ghost_weight = 0.8 + 0.5 * max(0, 4 - nearest_ghost) / 4
        if len(self.maze.pellets) < 20:
            ghost_weight += 0.2

        best_score = None
        best_region = 0
        best_cell = self.pacman.position
        for region, cells in enumerate(grouped):
            if not cells:
                continue
            count = len(cells)
            for cell in cells:
                distance = distances.get(cell)
                if distance is None:
                    continue
                ghost_distance = min(
                    self.maze.distance(cell, ghost.position)
                    for ghost in self.ghosts
                )
                score = distance - ghost_weight * ghost_distance - 0.03 * count
                if best_score is None or score < best_score:
                    best_score = score
                    best_region = region
                    best_cell = cell
        self._goal_cache = (distances.get(best_cell, 0), best_cell, best_region)
        return self._goal_cache

    def _ghost_transition_distribution(self, ghost, position, speed_progress,
                                       pacman_position, pacman_direction):
        """Exact one-turn ghost distribution under the original ghost rules."""
        next_progress = speed_progress + 70
        if next_progress < 100:
            return [(position, next_progress, 1.0)]
        next_progress -= 100

        legal = self.maze.legal_neighbors(position)
        if not legal:
            return [(position, next_progress, 1.0)]
        if ghost.name == "Chaser":
            target = pacman_position
        else:
            target = self._ambusher_prediction_from(
                pacman_position, pacman_direction, steps=3
            )
        follow_step = self.maze.next_step_towards(position, target)
        probabilities = {}
        if follow_step == position:
            for cell in legal:
                probabilities[cell] = probabilities.get(cell, 0.0) + 1.0 / len(legal)
        else:
            probabilities[follow_step] = probabilities.get(follow_step, 0.0) + ghost.target_rate
            random_share = (1.0 - ghost.target_rate) / len(legal)
            for cell in legal:
                probabilities[cell] = probabilities.get(cell, 0.0) + random_share
        return [(cell, next_progress, probability)
                for cell, probability in probabilities.items()]

    def _two_turn_safety_value(self, next_position, action, room_weight=2.0):
        """Expected best two-turn survival/escape score after a candidate move.

        This is a planning prior, not a change to ghost behavior. Each ghost's
        distribution uses its actual speed accumulator, target probability,
        shortest-path step, and random-neighbor fallback.
        """
        chaser_first = self._ghost_transition_distribution(
            self.chaser, self.chaser.position, self.chaser.speed_progress,
            next_position, action
        )
        ambusher_first = self._ghost_transition_distribution(
            self.ambusher, self.ambusher.position, self.ambusher.speed_progress,
            next_position, action
        )
        total_value = 0.0

        for (chaser_pos, chaser_phase, chaser_probability), \
                (ambusher_pos, ambusher_phase, ambusher_probability) in itertools.product(
                    chaser_first, ambusher_first
                ):
            probability = chaser_probability * ambusher_probability
            if chaser_pos == next_position or ambusher_pos == next_position:
                continue

            best_future = 0.0
            for next_action_index in self.legal_action_indices(next_position):
                next_action = ACTIONS[next_action_index]
                dx, dy = ACTION_VECTORS[next_action]
                second_position = (next_position[0] + dx,
                                   next_position[1] + dy)
                if second_position in (chaser_pos, ambusher_pos):
                    continue

                chaser_second = self._ghost_transition_distribution(
                    self.chaser, chaser_pos, chaser_phase,
                    second_position, next_action
                )
                ambusher_second = self._ghost_transition_distribution(
                    self.ambusher, ambusher_pos, ambusher_phase,
                    second_position, next_action
                )
                chaser_hit = sum(p for pos, _, p in chaser_second
                                 if pos == second_position)
                ambusher_hit = sum(p for pos, _, p in ambusher_second
                                   if pos == second_position)
                survival = (1.0 - chaser_hit) * (1.0 - ambusher_hit)

                # Reward open escape options after the second ghost response.
                room_value = 0.0
                for (cpos, _, cp), (apos, _, ap) in itertools.product(
                        chaser_second, ambusher_second):
                    minimum_distance = min(
                        self.maze.distance(second_position, cpos),
                        self.maze.distance(second_position, apos),
                    )
                    safe_exits = sum(
                        1 for neighbor in self.maze.legal_neighbors(second_position)
                        if self.maze.distance(neighbor, cpos) > 1
                        and self.maze.distance(neighbor, apos) > 1
                    )
                    room_value += cp * ap * (
                        min(5, minimum_distance) + 0.45 * safe_exits
                    )

                value = survival + room_weight * 0.035 * room_value
                if second_position in self.maze.pellets:
                    value += 0.001
                best_future = max(best_future, value)

            total_value += probability * best_future

        return total_value

    def action_prior_scores(self, legal_actions):
        """Return model-based action scores used as a light Q-learning prior."""
        goal_distance, goal_cell, _ = self._strategic_goal()
        first_step = self.maze.next_step_towards(self.pacman.position, goal_cell)
        scores = {}
        for action_index in legal_actions:
            action = ACTIONS[action_index]
            dx, dy = ACTION_VECTORS[action]
            destination = (self.pacman.position[0] + dx,
                           self.pacman.position[1] + dy)
            safety = self._two_turn_safety_value(destination, action, room_weight=2.0)
            score = 50.0 * safety - self.maze.distance(destination, goal_cell)
            if destination == first_step:
                score += 6.0
            if destination in self.maze.pellets:
                score += 1.0
            for ghost in self.ghosts:
                distance = self.maze.distance(destination, ghost.position)
                score -= 0.1 * {0: 100, 1: 20, 2: 5, 3: 1}.get(distance, 0)
            if action == self.pacman.direction:
                score += 2.0
            scores[action_index] = score
        return scores

    def observe(self):
        """Return a strategic state with local, regional, and threat context.

        State features:
        - exact Pac-Man tile and current heading;
        - 5x5 local pellet occupancy mask;
        - 5x5 regional occupancy mask and coarse global progress;
        - strategic pellet target direction, distance, and region;
        - exact relative ghost offsets, path distance, and movement phase;
        - legal-action mask.
        """
        px, py = self.pacman.position

        local_pellets = 0
        bit = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if (px + dx, py + dy) in self.maze.pellets:
                    local_pellets |= (1 << bit)
                bit += 1

        region_counts = self._region_counts()
        region_mask = 0
        for region, count in enumerate(region_counts):
            if count:
                region_mask |= (1 << region)

        collected = len(self.maze.floor) - len(self.maze.pellets)
        progress_bucket = min(10, (collected * 10) // max(1, len(self.maze.floor)))

        goal_distance, goal_cell, goal_region = self._strategic_goal()
        goal_first_step = self.maze.next_step_towards(self.pacman.position, goal_cell)
        goal_sector = self._relative_sector(self.pacman.position, goal_first_step)

        heading_id = ("UP", "RIGHT", "DOWN", "LEFT").index(self.pacman.direction)
        legal_mask = sum(1 << action for action in self.safe_action_indices())

        def ghost_features(ghost):
            dx = max(-7, min(7, ghost.position[0] - px))
            dy = max(-7, min(7, ghost.position[1] - py))
            distance = min(15, self.maze.distance(self.pacman.position,
                                                  ghost.position))
            # The accumulator cycles through ten exact phases for a 70% rate.
            speed_phase = ghost.speed_progress // 10
            return dx, dy, distance, speed_phase

        return (
            px, py, heading_id, local_pellets, region_mask, progress_bucket,
            goal_sector, min(20, goal_distance), goal_region,
            *ghost_features(self.chaser),
            *ghost_features(self.ambusher),
            legal_mask,
        )

    def _potential(self):
        """Potential for safe progress; used only for policy-invariant shaping."""
        if not self.maze.pellets:
            return 0.0
        goal_distance, _, _ = self._strategic_goal()
        chaser_distance = self.maze.distance(self.pacman.position,
                                             self.chaser.position)
        ambusher_distance = self.maze.distance(self.pacman.position,
                                               self.ambusher.position)
        pellet_potential = -0.55 * min(goal_distance, 12)
        safety_potential = 0.75 * (
            min(chaser_distance, 7) + min(ambusher_distance, 7)
        )
        return pellet_potential + safety_potential

    def _finish_with_learning(self, state, action_index, reward, terminal):
        self.agent.learn(state, action_index, reward, self.observe(), [], terminal)

    def step(self):
        """Advance Pac-Man and the unchanged 70%-speed ghost rules by one turn."""
        if self.terminal:
            return

        state = self.observe()
        legal_actions = self.safe_action_indices()
        action_bias = self.action_prior_scores(legal_actions)
        action_index = self.agent.choose_action(
            state, legal_actions, action_bias=action_bias
        )
        if action_index is None:
            self._finish_episode(False, "No legal move")
            return

        previous_potential = self._potential()
        previous_region_counts = self._region_counts()
        previous_remaining = len(self.maze.pellets)

        self.pacman.move(ACTIONS[action_index], self.maze)
        self.step_count += 1
        self._goal_cache = None

        collected = self.maze.consume_pellet(self.pacman.position)
        # A revisited, already-cleared tile is a real detour, not progress.
        reward = -0.08 + (9.0 + 3.0 * (1.0 - previous_remaining / max(1, len(self.maze.floor)))
                          if collected else -1.1)

        if collected:
            region = self._region_id(self.pacman.position)
            if previous_region_counts[region] == 1:
                reward += 8.0  # Explicit signal for finishing a 3x3 sub-region.

        # Collision is checked before ghost movement, just as in the original game.
        if any(ghost.position == self.pacman.position for ghost in self.ghosts):
            self._finish_with_learning(state, action_index, reward - 250.0, True)
            self._finish_episode(False, "Caught by a ghost")
            return

        # Ghost movement, target selection, randomness, and speed are unchanged.
        for ghost in self.ghosts:
            ghost.move(self.maze, self.pacman)
            self._goal_cache = None
            if ghost.position == self.pacman.position:
                self._finish_with_learning(state, action_index, reward - 250.0, True)
                self._finish_episode(False, "Caught by a ghost")
                return

        if not self.maze.pellets:
            reward += 300.0
            self._finish_with_learning(state, action_index, reward, True)
            self._finish_episode(True, "Maze cleared!")
            return

        if self.step_count >= MAX_STEPS_PER_EPISODE:
            self._finish_with_learning(state, action_index, reward - 100.0, True)
            self._finish_episode(False, "Step limit reached")
            return

        # Potential-based reward shaping: progress to a strategic pellet cluster
        # and maintaining distance from both ghosts. Gamma matches the learner.
        next_potential = self._potential()
        reward += self.agent.gamma * next_potential - previous_potential

        # Small immediate warning for a very close ghost, in addition to shaping.
        nearest_ghost = min(
            self.maze.distance(self.pacman.position, ghost.position)
            for ghost in self.ghosts
        )
        if nearest_ghost == 1:
            reward -= 1.5
        elif nearest_ghost == 2:
            reward -= 0.5

        next_state = self.observe()
        next_legal = self.safe_action_indices()
        self.agent.learn(state, action_index, reward, next_state, next_legal)

    def _finish_episode(self, won, result):
        if self.terminal:
            return
        self.terminal = True
        self.completed_episodes += 1
        if won:
            self.wins += 1
        self.episode_history.append(bool(won))
        self.episode_history = self.episode_history[-20:]
        self.last_result = result
        self.agent.end_episode()
        self.terminal_timer = 1.1

    def update(self, delta_seconds):
        """Handle the short pause between episodes. Movement timing is in main.py."""
        if self.terminal:
            self.terminal_timer -= delta_seconds
            if self.terminal_timer <= 0:
                self.episode += 1
                self.reset_episode()

    @property
    def recent_win_rate(self):
        """Win rate over the most recent 20 completed episodes."""
        if not self.episode_history:
            return 0.0
        return 100.0 * sum(self.episode_history) / len(self.episode_history)

    @property
    def win_rate(self):
        if self.completed_episodes == 0:
            return 0.0
        return 100.0 * self.wins / self.completed_episodes
