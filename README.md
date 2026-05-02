# Position Prediction Obstacle Field

This is a playable Pygame obstacle field with continuous 2D movement, wall collisions, a position-only trace, raycast prediction visuals, AI training agents, and an online movement predictor that trains while you play.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

The game implementation lives in `src/main.py`; the root `main.py` is a small launcher.

Movement uses a fixed `1/120` second physics step with input acceleration and linear drag. Terminal speeds are `120 px/s` crouching, `260 px/s` walking, and `380 px/s` sprinting.

The predictor trains online while you play. Every 24 Hz sampled position creates a supervised training example once enough future samples arrive. The HUD shows collected samples, optimizer updates, replay buffer size, and the moving-average loss.

AI training agents sometimes idle. When moving, each behavior gets an evenly distributed random terminal speed from very slow movement up to `3x` player sprint speed. Their paths vary between pauses, straight runs, smooth arcs, wandering turns, jittery short corrections, and occasional rapid tap-strafe bursts.

Prediction visuals are bounded to the model's knowledge window: from the next available model step to `0.5` seconds ahead.

The predictor outputs multiple candidate futures rather than one averaged path. Training uses a best-candidate rollout loss, weights the first `0.25` seconds most heavily, applies medium weight from `0.25` to `0.5` seconds, ignores longer horizons, penalizes wall-crossing line segments, and lightly discourages movement predictions when the target stayed still.

Training persists in `checkpoints/`:

- `movement_model.pt` stores model weights, optimizer state, and training counters.
- `training_data.pt` stores the replay buffer of collected training samples.

The game loads these files on startup, autosaves every 720 optimizer updates, and saves again when it exits.

Controls:

- Move with `WASD` or arrow keys.
- Hold `Ctrl` to crouch.
- Hold `Shift` to sprint.
- Press `1` to toggle the blue recent-movement path.
- Press `2` to toggle the orange prediction path.
- Press `3` to toggle yellow raycast lines.
- Press `4` to toggle AI training agents.
- Press `R` to reset.
- Press `Esc` to quit.
