# HumanTrace-Lightweight-Human-to-Robot-Trajectory-Retargeting

HumanTrace is a lightweight human-to-robot manipulation retargeting pipeline developed for the Humanoid Robot Learning Research Internship Challenge.

The project investigates a simple question:
> Can a small amount of personally collected human manipulation data provide useful motion guidance for a robot without training a large policy model?

Instead of directly training a VLA or imitation-learning model, the current prototype extracts the geometric structure of a human manipulation trajectory and transfers it to a Panda robot in the LIBERO simulation environment.

---

## Overview

The pipeline is:

```text
Personally collected phone video
        ↓
Manual object trajectory annotation
        ↓
Transport-phase segmentation
        ↓
Savitzky-Golay smoothing
        ↓
Relative trajectory representation
        ↓
Arc-length resampling
        ↓
Human-to-robot workspace scaling
        ↓
OSC end-effector controller
        ↓
Panda robot in LIBERO
```

The key design idea is to separate:

- **human motion intent** — where the manipulated object should travel;
- **robot embodiment-specific execution** — how the Panda arm approaches, grasps, lifts and releases the object.

The human demonstration therefore provides the motion geometry rather than directly specifying robot joint commands.

---

## Personally Collected Data

A short tabletop manipulation demonstration was recorded using a phone.

The demonstrated object was manually moved along a curved trajectory rather than directly between its start and goal locations. This provides a simple setting for comparing a human-guided path against a geometry-only straight-line baseline.

The original demonstration is converted into a sequence of manually labelled 2D object positions.

The current example contains:

- a manually identified transport phase;
- approximately 40 transport trajectory samples;
- a smoothed representation;
- 20 spatially resampled robot waypoints.

A compressed example of the personally collected demonstration is included in `data/samples/`.

Processed trajectory data is provided in `data/processed/`.

---

## Trajectory Processing

### 1. Phase segmentation

The initial grasping period and final release/adjustment period are excluded from the transferred trajectory.

Only the object transport phase is used to define robot motion.

### 2. Smoothing

Manual image annotations contain small localisation errors and human motion jitter.

A Savitzky-Golay filter is used to smooth the trajectory while approximately preserving its curved geometry.

### 3. Relative representation

Absolute image coordinates are not directly transferred to the robot.

Instead, each trajectory is represented relative to its initial transport position:

\[
\Delta p_t = p_t - p_0
\]

This makes the representation less dependent on the camera placement and absolute human workspace.

### 4. Normalisation

The trajectory is normalised while preserving its two-dimensional shape.

### 5. Arc-length resampling

The processed path is resampled into 20 approximately spatially uniform waypoints.

This avoids directly reproducing variations in human movement speed and gives the robot controller a cleaner geometric path.

---

## Human-to-Robot Retargeting

The normalised human trajectory is scaled into a safe Panda end-effector workspace:

\[
p_t^{robot}
=
p_0^{robot}
+
S \Delta p_t^{human}
\]

where \(S\) is a workspace scaling factor.

Only planar motion is currently transferred.

Vertical motion and grasp/release behaviour are deliberately separated from the demonstration trajectory and will be handled by robot-specific control logic.

---

## Robot Control

Experiments use the Panda robot in a LIBERO spatial manipulation environment.

The end effector is controlled using LIBERO / robosuite's OSC pose controller.

For each human-derived waypoint, a proportional controller calculates the end-effector position error and generates bounded Cartesian control commands.

---

## Current Result

The current human demonstration was successfully transferred to the Panda end effector as a sequence of 20 waypoints.

For the current test:

- maximum transferred planar displacement: approximately **0.10 m**;
- all 20 human-derived waypoints were executed;
- final end-effector target error: approximately **7.9 mm**.

This establishes an end-to-end prototype:

```text
real human data
→ trajectory extraction
→ trajectory processing
→ retargeting
→ simulated robot execution
```

---

## Baseline

A straight-line baseline is evaluated using:

- the same robot;
- the same start pose;
- the same endpoint;
- the same number of waypoints;
- the same controller.

The only difference is trajectory geometry.

The baseline path is defined by linear interpolation:

\[
p(t) = (1-t)p_{start} + t p_{goal}
\]

This allows the effect of human-provided motion geometry to be evaluated independently of the robot controller.

---

## Repository Structure

```text
scripts/
    label_trajectory.py
    process_trajectory.py
    run_human_trajectory.py
    run_straight_baseline.py

data/
    processed/
    samples/

results/
    figures/
    examples/

docs/
```

---

## Setup

The project was tested with:

- Ubuntu 22.04 via WSL2
- Python 3.8.13
- LIBERO
- robosuite 1.4.0
- MuJoCo

Create a Python environment:

```bash
conda create -n humanoid_libero python=3.8.13
conda activate humanoid_libero
```

Clone and install LIBERO following the official repository instructions.

Then install the additional processing dependencies required by this project.

---

## Running the Pipeline

### Process the human demonstration

```bash
python scripts/process_trajectory.py
```

This produces the cleaned trajectory and robot-ready waypoints.

### Run the human-guided robot trajectory

```bash
python scripts/run_human_trajectory.py
```

### Run the straight-line baseline

```bash
python scripts/run_straight_baseline.py
```

---

## What Worked

So far:

- relative trajectory transfer was more appropriate than attempting to map absolute image coordinates;
- trajectory smoothing reduced annotation jitter while retaining the overall path shape;
- arc-length resampling produced a compact and consistent robot trajectory;
- a simple OSC proportional controller reliably followed all 20 human-derived waypoints;
- separating human motion intent from robot-specific execution substantially simplified the implementation.

---

## What Did Not Work / Limitations

Several limitations remain in the current prototype:

- the manipulation trajectory is manually annotated rather than automatically tracked;
- transport-phase segmentation is currently manual;
- monocular depth is not transferred;
- the current prototype validates end-effector trajectory retargeting but does not yet complete the full grasp-transfer-place sequence;
- human-to-robot workspace scaling is manually selected;
- a single demonstration is currently used for the initial prototype.

These choices were deliberate in the first iteration to isolate and validate the core retargeting problem before adding additional perception and manipulation complexity.

---

## Next Steps

The next experiments are:

1. compare human-guided trajectories against the straight-line baseline;
2. add automatic grasp, lift, transport and release phases;
3. evaluate obstacle-sensitive trajectories;
4. compare raw and smoothed demonstrations;
5. test robustness to trajectory noise and workspace scaling;
6. optionally replace manual annotation with automatic visual tracking.

---

## Motivation

This project deliberately prioritises implementation simplicity and interpretable experimentation over model complexity.

Rather than assuming a large learned policy is always necessary, the prototype explores how far carefully processed human motion priors can be transferred across embodiment using a lightweight retargeting pipeline.
