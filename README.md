# Pac-Man Q-Learning (Python + Pygame)

A modular 15 × 15 Pac-Man project using tabular Q-learning with a compact
model-based action prior. The included `q_table.json` is trained and evaluated
against the exact maze and ghost rules described below.

## Fixed game rules

- The maze is the exact 15 × 15 `MAZE_LAYOUT` in `maze.py` (111 walkable cells).
- Every walkable cell starts with one pellet. Pac-Man's starting pellet is
  consumed when an episode starts.
- Chaser follows Pac-Man directly with target-seeking probability **0.8**.
- Ambusher targets a position **three tiles ahead** with target-seeking
  probability **0.6**.
- Both ghosts move at **70%** of Pac-Man's tile-step rate (7 ghost moves per
  10 Pac-Man turns).
- One life per episode. Contact with either ghost ends that episode.

The maze layout and `ghost.py` movement/target rules were not changed for this
learning improvement.

## What changed in the learner

### State representation (18 fields)

The old state aliased very different board situations. The new state contains:

1. Pac-Man's exact tile and heading.
2. A packed **5 × 5 local pellet mask** around Pac-Man.
3. A packed **5 × 5 macro-region occupancy mask** showing which areas still
   contain pellets.
4. Global collection progress, strategic target direction/distance/region.
5. For each ghost: relative x/y offset, maze-path distance, and its 70%-speed
   accumulator phase.
6. A bit mask of currently safe legal actions.

This gives the Q-table both local detail and enough global context to avoid
repeatedly circling a small area while large pellet regions remain.

### Reward design

- Collecting a new pellet: **+9**, rising with overall maze progress.
- Revisit a cleared tile: **−1.1** (plus the normal step cost).
- Finish the last pellet in a 3 × 3 region: **+8**.
- Being caught: **−250**.
- Clear the maze: **+300**.
- Reaching the episode step limit: **−100**.
- Potential-based shaping rewards moving toward a strategic remaining-pellet
  goal while retaining distance from both ghosts; the shaping uses the same
  discount factor as Q-learning.
- A small immediate penalty discourages ending a turn adjacent to a ghost.

### Safety-aware action prior

The maze is small enough to model a short ghost horizon. Before selecting an
action, the agent calculates a two-turn expected survival/escape score using
both ghosts' existing target probabilities, shortest-path moves, random legal
fallbacks, and exact 70%-speed phases. This is only an action-selection prior:
`ghost.py` still controls all actual ghost moves. High immediate capture-risk
moves are filtered from the learner's candidate actions where a safer legal
move exists. Q-values are still updated after every move using the standard
Q-learning update; the learned Q-values break close ties in action selection.

## Included Q-table and measured result

`q_table.json` contains **308,081 learned states** (state schema version 4).
The retained checkpoint was trained for 2,000 episodes. Continued training with
exploration reheating was tested for another 600 episodes, but its candidate
checkpoint did not outperform the original on the final held-out seed set.

### Independent frozen-policy evaluation (target not reached)

Evaluation uses epsilon = 0 and disables Q-value updates. The retained policy
was tested on five held-out ghost-randomness seeds, 50 episodes per seed:

| Seed | Wins | Episodes | Win rate |
|---:|---:|---:|---:|
| 95001 | 33 | 50 | 66.0% |
| 95002 | 35 | 50 | 70.0% |
| 95003 | 40 | 50 | 80.0% |
| 95004 | 36 | 50 | 72.0% |
| 95005 | 41 | 50 | 82.0% |
| **Combined** | **185** | **250** | **74.0%** |

The requested **greater-than-90%** win rate has **not** been achieved. The
continued-training candidate scored 72.0% on a separate 250-episode evaluation
and was rejected in favor of the original checkpoint. These are empirical
measurements, not a guarantee of future performance.

Reproduce the independent evaluation:

```bash
python evaluate.py --qtable q_table.json --episodes 50 --seeds 95001 95002 95003 95004 95005
```

Resume training. Exploration is reheated to 0.25 by default, and the existing
checkpoint is evaluated first; it is replaced only when a later validation
result is strictly better. With `--until-target`, training continues until
validation is strictly above the target (interrupting safely leaves the last
best checkpoint intact):

```bash
python train.py --until-target --eval-episodes 100 --check-every 100 --target 90 --resume
```

Use `--resume-epsilon` to change the exploration reheat level and
`--eval-seeds` to select validation seeds. Keep final evaluation seeds separate
from training-validation seeds.

## Install and run the Pygame UI

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment and install Pygame:

```bash
# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

`main.py` automatically loads `q_table.json` if it exists, then continues
learning. The UI keeps the compact metrics requested earlier: episode, learned
states, win rate, remaining pellets, and episode steps as a number. UP raises
training speed with no upper cap; DOWN lowers it (minimum 0.1×). Press `S` to
save manually. Autosave occurs every 100 completed episodes, and the Q-table
is saved again when the game closes.

## Controls

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `UP` | Increase speed by 0.5× (no upper cap) |
| `DOWN` | Decrease speed by 0.5× (minimum 0.1×) |
| `R` | Start a fresh episode |
| `S` | Save Q-table |
| `ESC` | Quit and save |

## Project structure

```text
pacman_qlearning_project/
├── main.py             # Pygame UI and controls
├── game.py             # State, reward, action prior, episode loop
├── maze.py             # Exact 15 × 15 maze and pathfinding
├── pacman.py           # Pac-Man movement and ambusher prediction
├── ghost.py            # Original Chaser and Ambusher rules
├── agent.py            # Tabular Q-learning and JSON persistence
├── autosave.py         # Every-100-completed-episodes autosave
├── evaluate.py         # Frozen-policy evaluation
├── train.py            # Resumable training and threshold checks
├── q_table.json        # Trained state-version-4 Q-table
├── config.py           # Shared settings
└── tests/              # Unit tests
```

## Tests

```bash
python -m unittest discover -s tests -v
```
