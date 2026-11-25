"""Maze layout, pellet management, and simple breadth-first pathfinding."""

from collections import deque


# A hand-designed 15x15 maze. Open spaces become pellets when an episode starts.
# The corridors include loops, dead ends, and several connecting passages.
MAZE_LAYOUT = (
    "###############",
    "#.............#",
    "#.###.###.###.#",
    "#.........#...#",
    "#.#.#.###.#.#.#",
    "#...#...#...#.#",
    "#.###.#.#.###.#",
    "#.....#.#.....#",
    "#.###.#.#.###.#",
    "#.#...#.......#",
    "#.#.#.###.#.###",
    "#...#.........#",
    "#.###.###.###.#",
    "#.............#",
    "###############",
)


class Maze:
    """Static maze geometry plus the pellets remaining in the current episode."""

    def __init__(self, layout=MAZE_LAYOUT):
        if len(layout) != 15 or any(len(row) != 15 for row in layout):
            raise ValueError("The maze must be exactly 15 rows by 15 columns.")

        self.layout = tuple(layout)
        self.width = len(layout[0])
        self.height = len(layout)
        self.walls = set()
        self.floor = set()

        for y, row in enumerate(layout):
            for x, character in enumerate(row):
                if character == "#":
                    self.walls.add((x, y))
                elif character == ".":
                    self.floor.add((x, y))
                else:
                    raise ValueError(f"Unexpected maze character: {character!r}")

        self._distances = {}
        self._first_steps = {}
        self._precompute_paths()
        self.reset_pellets()

    def _precompute_paths(self):
        """Precompute shortest distances and first steps for this small static maze."""
        for source in self.floor:
            queue = deque([source])
            distances = {source: 0}
            first_steps = {source: source}
            while queue:
                current = queue.popleft()
                for neighbor in self.legal_neighbors(current):
                    if neighbor in distances:
                        continue
                    distances[neighbor] = distances[current] + 1
                    first_steps[neighbor] = (
                        neighbor if current == source else first_steps[current]
                    )
                    queue.append(neighbor)
            self._distances[source] = distances
            self._first_steps[source] = first_steps

    def reset_pellets(self):
        """Put one pellet on every walkable cell."""
        self.pellets = set(self.floor)

    def in_bounds(self, position):
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, position):
        return position in self.floor

    def legal_neighbors(self, position):
        x, y = position
        candidates = ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))
        return [cell for cell in candidates if self.is_walkable(cell)]

    def consume_pellet(self, position):
        """Remove and report a pellet at position, if one is still there."""
        if position in self.pellets:
            self.pellets.remove(position)
            return True
        return False

    def shortest_path(self, start, goal):
        """Return a shortest path (including start and goal), or [] if unreachable."""
        if start == goal:
            return [start]
        if start not in self._distances or goal not in self._distances[start]:
            return []

        path = [start]
        current = start
        while current != goal:
            current = self._first_steps[current][goal]
            path.append(current)
        return path

    def distance(self, start, goal):
        """Return shortest-path distance, or a large value if unreachable."""
        return self._distances.get(start, {}).get(goal, self.width * self.height + 1)

    def next_step_towards(self, start, goal):
        if start == goal:
            return start
        return self._first_steps.get(start, {}).get(goal, start)
