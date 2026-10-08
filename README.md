# HumanTrace: Lightweight Human-to-Robot Trajectory Retargeting

HumanTrace is a lightweight pipeline for transferring the **geometric structure of a personally collected human manipulation demonstration** to a Panda robot in LIBERO.

The project asks a simple question:

> Can a small amount of personally collected human motion data directly shape robot behaviour without training a large imitation-learning policy?

In this prototype, the answer is yes.

A short smartphone demonstration is manually annotated to recover the manipulated object's trajectory. The trajectory is then smoothed, normalized, resampled, and geometrically retargeted to the robot workspace.

The robot performs a complete manipulation sequence:

**grasp → lift → human-derived transport → place → release**

while preserving the curved transport geometry of the human demonstration.

---

## Demo

![HumanTrace Pick-and-Place](results/humantrace_pick_place.gif)

The final HumanTrace run successfully completes the LIBERO task:

> **Pick up the black bowl from the table centre and place it on the plate.**

Final result:

| Metric | Result |
|---|---:|
| LIBERO reward | **1.0** |
| Task completed | **True** |
| Final bowl-to-plate XY error | **5.4 mm** |

---

## Core Idea

HumanTrace separates manipulation into two parts:

```text
Robot-specific interaction
        +
Human-derived transport geometry
```

The Panda-specific primitive handles the embodiment-dependent interaction:

```text
approach → grasp → lift
```

The personally collected human demonstration determines:

```text
the geometry of the transport trajectory
```

Placement and release are then handled by a simple robot-specific controller.

This separation is intentional.

Grasping depends strongly on gripper geometry, end-effector orientation, contact dynamics, and robot embodiment. In contrast, the overall geometric structure of a demonstrated transport motion can be transferred between embodiments.

The key contribution of the personally collected data is therefore explicit:

> **the human demonstration directly changes the path followed by the robot while transporting the grasped object.**

---

## Pipeline

```text
Personally collected smartphone video
                ↓
Manual object-centre annotation
                ↓
Savitzky-Golay smoothing
                ↓
Relative trajectory representation
                ↓
Trajectory normalization
                ↓
Arc-length resampling
                ↓
20 human waypoints
                ↓
2D similarity transform
(rotation + scale + translation)
                ↓
Panda transport trajectory
                ↓
Complete LIBERO pick-and-place
```

---

## 1. Personally Collected Human Demonstration

The source manipulation demonstration was personally recorded using a smartphone.

The demonstrator moves an object along a deliberately curved path rather than taking the shortest direct route from start to goal.

The raw video is not included in the public repository in order to keep the submission lightweight. It is excluded through `.gitignore`.

All trajectory data derived from the recording is included under:

```text
data/processed/
```

The demonstration was manually annotated, producing **57 labelled object-centre observations**.

The transport phase was then isolated and processed into a compact robot-transfer representation.

---

## 2. Human Trajectory Processing

The manually labelled image-space trajectory contains small annotation noise and non-uniform temporal spacing.

The processing pipeline therefore performs several operations.

### Raw trajectory and smoothing

A Savitzky-Golay filter is applied to reduce annotation noise while preserving the overall shape of the demonstrated motion.

![Raw vs Smoothed Human Trajectory](results/human_trajectory_raw_vs_smooth.png)

The trajectory is then converted into relative displacement from the beginning of the selected transport phase.

The image-space vertical direction is flipped so that the representation follows a Cartesian-style coordinate convention.

The resulting trajectory is normalized while preserving its geometric shape.

### Normalized trajectory

![Normalized Human Trajectory](results/human_trajectory_normalized.png)

Finally, arc-length resampling converts the trajectory into **20 approximately equally spaced waypoints**.

This produces a lightweight motion representation that can be transferred to a robot with a different workspace size and orientation.

---

## 3. Human-to-Robot Retargeting

Let the processed human trajectory be

```text
p_h(t)
```

The robot transport trajectory is generated using a 2D similarity transform:

```text
p_r(t) = p_start + s R p_h(t)
```

where:

- `R` rotates the human start-to-end direction toward the robot bowl-to-plate direction,
- `s` scales the demonstrated displacement to the robot task distance,
- `p_start` translates the transformed trajectory to the robot's transport start position.

This transformation preserves the **shape of the human motion** while adapting it to the robot workspace.

For the successful HumanTrace run:

```text
Human trajectory endpoint:
[-0.8452, 0.4393]

Robot bowl → plate vector:
[0.1432, 0.1773]

Rotation:
-101.46 degrees

Scale:
0.2393
```

The transformed trajectory therefore ends at the correct planar task displacement while retaining the non-linear structure of the human demonstration.

---

## 4. Robot-Specific Grasp Primitive

Trajectory following was relatively straightforward, but stable object grasping proved much more embodiment-dependent.

An initial position-only approach was tested using a brute-force search across **48 candidate grasp positions** around the bowl rim.

None produced a reliable lift.

This indicated that successful grasping depends on more than Cartesian position alone, including:

- end-effector orientation,
- contact geometry,
- gripper timing,
- interaction dynamics,
- robot embodiment.

To keep the project focused on human-to-robot trajectory transfer rather than grasp planning, the final system uses a compact Panda-specific grasp-and-lift primitive extracted from a successful LIBERO demonstration.

Only the information required for the primitive is retained:

```text
one simulator initial state
+
56 grasp-and-lift actions
```

The resulting file is:

```text
data/robot_primitives/panda_grasp_demo0.npz
```

and is only a few kilobytes in size.

Importantly:

> **The LIBERO primitive is used for embodiment-specific grasp and lift behaviour only. The transport path after grasping is generated from the personally collected human demonstration.**

The full official LIBERO demonstration dataset is not required by the final HumanTrace repository.

---

## 5. Complete HumanTrace Pick-and-Place

The final execution follows this sequence:

```text
Initial task state
        ↓
Panda grasp primitive
        ↓
Stable bowl lift
        ↓
Human trajectory retargeting
        ↓
20-waypoint curved transport
        ↓
Align bowl above plate
        ↓
Descend
        ↓
Open gripper
        ↓
Retract
        ↓
Task success
```

The final HumanTrace execution achieved:

```text
Final bowl-to-plate XY error: 0.0054 m
LIBERO reward:               1.0
Environment done:            True
```

Example result files are stored in:

```text
results/final_pick_place/
```

This directory corresponds to the **HumanTrace human-derived transport experiment**.

---

## 6. Straight-Line Baseline

A controlled straight-line baseline is included to show that the personally collected human trajectory has a real and measurable effect on robot behaviour.

The HumanTrace and straight-line experiments use the same:

- initial simulator state,
- Panda grasp primitive,
- controller parameters,
- number of transport waypoints,
- transport start,
- destination,
- alignment procedure,
- placement procedure,
- release procedure.

The only experimental difference is the transport geometry.

```text
HumanTrace:
human-derived curved trajectory

Straight-line baseline:
linear interpolation from start to goal
```

The corresponding result directories are:

```text
results/final_pick_place/
```

for the **HumanTrace experiment**, and

```text
results/final_pick_place_straight/
```

for the **straight-line baseline**.

Both are complete pick-and-place experiments.

---

## 7. Quantitative Comparison

![HumanTrace vs Straight-Line Baseline](results/humantrace_vs_straight.png)

| Metric | HumanTrace | Straight Line |
|---|---:|---:|
| Transport waypoints | 20 | 20 |
| Target path length | **0.414 m** | **0.228 m** |
| Maximum lateral deviation | **0.146 m** | **~0 m** |
| Mean tracking error | **7.3 mm** | **7.4 mm** |
| Final placement error | **5.4 mm** | **5.5 mm** |
| Task success | **Yes** | **Yes** |

The straight-line baseline is naturally shorter.

HumanTrace is **not** intended to outperform a straight line in path length. A straight line is expected to be the geometrically shortest route between the same endpoints.

Instead, the experiment tests whether the non-linear structure of a human motion can be transferred to a robot **without sacrificing successful task execution**.

The largest lateral deviation from the direct start-to-goal line was:

```text
HumanTrace:    146 mm
Straight line: ~0 mm
```

This shows that the human demonstration produced a substantial and measurable change in robot transport geometry.

At the same time, controller tracking remained almost identical:

```text
HumanTrace mean tracking error:    7.3 mm
Straight-line mean tracking error: 7.4 mm
```

and final placement accuracy was also essentially unchanged:

```text
HumanTrace final placement error:    5.4 mm
Straight-line final placement error: 5.5 mm
```

Both approaches completed the LIBERO task successfully.

---

## Main Result

The purpose of HumanTrace is not to generate the shortest possible trajectory.

The result is instead:

> **A personally collected human manipulation demonstration can directly and measurably alter the geometry of robot behaviour while preserving successful task execution.**

The robot does not simply collapse the manipulation into a direct start-to-goal trajectory.

It retains the non-linear geometric structure of the demonstrated human motion.

---

## Repository Structure

```text
HumanTrace/
│
├── README.md
├── .gitignore
│
├── data/
│   ├── processed/
│   │   ├── demo01_points.csv
│   │   ├── demo01_clean.csv
│   │   └── demo01_waypoints.csv
│   │
│   └── robot_primitives/
│       └── panda_grasp_demo0.npz
│
├── scripts/
│   ├── label_trajectory.py
│   ├── process_trajectory.py
│   ├── run_human_trajectory.py
│   ├── run_straight_baseline.py
│   ├── run_pick_place.py
│   ├── run_pick_place_straight.py
│   ├── analyze_results.py
│   └── make_demo_gif.py
│
└── results/
    ├── comparison_metrics.csv
    ├── human_trajectory_raw_vs_smooth.png
    ├── human_trajectory_normalized.png
    ├── humantrace_vs_straight.png
    ├── humantrace_pick_place.gif
    │
    ├── final_pick_place/
    │   ├── 01_after_grasp.png
    │   ├── transport_10.png
    │   ├── 23_final.png
    │   └── transport_log.csv
    │
    └── final_pick_place_straight/
        ├── transport_19.png
        ├── 23_final.png
        └── transport_log.csv
```

---

## Running the Project

The project was developed with:

```text
Python 3.8
LIBERO
robosuite
MuJoCo
NumPy
Pandas
SciPy
OpenCV
Matplotlib
imageio
```

A working LIBERO installation is required.

Activate the environment and enter the project:

```bash
conda activate humanoid_libero
cd ~/humanoid_challenge/humantrace
```

### Process the human trajectory

```bash
python scripts/process_trajectory.py
```

### Run the HumanTrace pick-and-place experiment

```bash
python scripts/run_pick_place.py
```

A successful run should end with:

```text
Environment reward: 1.0
Environment done: True
```

### Run the straight-line baseline

```bash
python scripts/run_pick_place_straight.py
```

### Reproduce the quantitative comparison

```bash
python scripts/analyze_results.py
```

This generates:

```text
results/comparison_metrics.csv
results/humantrace_vs_straight.png
```

### Generate the demonstration GIF

```bash
python scripts/make_demo_gif.py
```

---

## Additional Validation Scripts

Two earlier trajectory-only experiments are also retained:

```text
scripts/run_human_trajectory.py
scripts/run_straight_baseline.py
```

These validate trajectory tracking independently of grasp and placement.

The main final evaluation uses:

```text
scripts/run_pick_place.py
scripts/run_pick_place_straight.py
```

---

## What Worked

Manual trajectory annotation proved sufficient for a lightweight prototype.

Savitzky-Golay smoothing reduced annotation noise without removing the overall demonstrated motion shape.

Arc-length resampling produced stable waypoint spacing.

The 2D similarity transform successfully adapted the human trajectory to a different robot workspace.

The Panda maintained a stable grasp while following the curved human-derived path.

Finally, the complete LIBERO manipulation task was successfully completed.

---

## What Did Not Work

A purely position-based grasp strategy was not sufficient.

Even after testing 48 candidate Cartesian grasp positions around the bowl rim, the robot did not achieve a reliable lift.

This failure highlighted an important distinction between:

```text
embodiment-specific interaction
```

and

```text
transferable motion geometry
```

The final system explicitly separates these two components.

---

## Limitations

This prototype currently uses:

- one personally collected human demonstration,
- manual rather than automatic trajectory extraction,
- planar 2D trajectory retargeting,
- one LIBERO manipulation task,
- one Panda-specific grasp primitive,
- no learned grasp policy,
- no depth reconstruction from the smartphone video.

The current result therefore demonstrates feasibility rather than broad generalization.

---

## Future Work

Natural extensions include:

- automatic object tracking from RGB video,
- multiple human demonstrations,
- 3D trajectory reconstruction,
- obstacle-aware trajectory transfer,
- evaluation across multiple LIBERO tasks,
- learned grasp primitives,
- adaptation across different robot embodiments.

---

## Key Takeaway

HumanTrace shows that useful human-to-robot transfer does not necessarily require a large learned policy.

A single personally collected human manipulation demonstration was sufficient to produce a visibly and quantitatively different robot transport trajectory while still completing the manipulation task successfully.
