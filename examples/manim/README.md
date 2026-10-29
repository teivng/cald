# The CALD animation

Source for `docs/cald_story.gif` / `docs/cald_story.mp4` — the method on a 2-D
task with the real TabICLv2 backbone. Every number in the animation (the
probability fields, boundaries, anchor-loss trajectories, the K = 50
calibration record, both verdicts) comes from actual CALD runs; the scene only
replays recorded data.

Two stages:

```bash
# 1. run CALD and dump fields/trajectories/record into assets/   (GPU, ~10 min)
python gen_data.py

# 2. render the scene (no GPU or torch needed; manim CE + ffmpeg)
manim -qm --disable_caching scene.py CALDStory
```

The harmful deployment batch is the benign one rotated 90° clockwise — the
paper's controlled 2-D experiment (rotation → harmful shift).
