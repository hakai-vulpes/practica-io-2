# Intelligent Optimization Practice 2

This repository contains two main components developed for the Operational Research practice: **Reinforcement Learning for Racetrack** and **Image Morphing using Optimal Transport**.

> Disclaimer: README is generated with AI assistance.

## Project Structure

- **`episodic-racing/`**: Contains the Reinforcement Learning implementation.
  - `racetrack_env.py`: Custom environment for the Racetrack problem.
  - `mc_agent.py`: Monte Carlo Control agent (Epsilon-Greedy).
  - `main.ipynb`: Notebook for training and visualizing the agent.
- **`face-morphing/`**: Contains the Image Morphing project.
  - `morphing.ipynb`: Todo.
  - `images/`: Directory for input images used in morphing with several examples.

## Prerequisites

This project is managed with `uv`. The dependencies are defined in `pyproject.toml` and include:
- `numpy`, `matplotlib`, `scipy`
- `opencv-python`, `pillow`, `scikit-image`
- `pot` (Python Optimal Transport)

## Installation

Ensure you have Python 3.12+ installed.

### Using uv (Recommended)

```bash
uv sync
```

### Manual Installation

If you prefer pip, you can install the dependencies manually:

```bash
pip install numpy matplotlib scipy pot opencv-python pillow scikit-image ipykernel
```

## Usage

### Part 1: Racetrack (Reinforcement Learning)

1. Navigate to the `episodic-racing` directory.
2. Open and run `main.ipynb` to train the Monte Carlo agent on the racetrack environment.
   - The agent learns to navigate from 'S' (Start) to 'F' (Finish) while avoiding '#' (Walls).
   - The implementation uses First-Visit Monte Carlo with Epsilon-Greedy exploration.

### Part 2: Image Morphing (Optimal Transport)

1. Navigate to the `face-morphing` directory.
2. Place your input images in the `images/` directory.
3. Open `morphing.ipynb`.
4. The functionality is the following:
   - Detects and aligns faces from the input images.
   - Computes the optimal transport plan between two images to generate smooth morphing transitions.
   - Displays the original images and the intermediate morphed steps.

## Guidelines & Details

- **RL Agent**: Uses a Q-table `(y, x, vy, vx)` to store state-action values. It learns essentially by trial and error, propagating rewards back from the finish line.
- **Morphing**: Uses the Sinkhorn algorithm to compute optimal transport plans, allowing for geometrically meaningful interpolations between images (unlike simple pixel blending).
