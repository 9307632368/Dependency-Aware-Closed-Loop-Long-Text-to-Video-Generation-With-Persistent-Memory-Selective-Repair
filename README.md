# Dependency-Aware Closed-Loop Long Text-to-Video Generation

---

## Overview

This project focuses on generating long videos from a single text prompt. Most existing text-to-video models work well only for short clips, but they struggle when the input becomes long and contains multiple scenes.

In such cases, problems like identity change, wrong locations, or broken continuity often appear. The goal of this work is to reduce these issues by introducing a structured generation pipeline instead of treating the entire prompt as a single input.

---

## Idea

Instead of generating the whole video at once, the prompt is first divided into smaller scenes. Then each scene is generated step by step while keeping track of what has already appeared in previous scenes.

The important part is that scenes are not independent. Some scenes depend on earlier ones (for example, same character or same location), so this dependency is explicitly modeled.

---

## Key Approach

### Scene Segmentation

The input prompt is divided into multiple smaller parts (scenes). This makes long prompts easier to handle.

### Dependency Handling

Each scene is checked to see whether it depends on previous scenes. This helps in maintaining consistency across the video.

### Scene-wise Generation

Instead of one large generation, each scene is generated separately using video diffusion models like:

* Stable Video Diffusion
* CogVideoX

### Continuity Handling

Important frames (keyframes) are taken from previous scenes and used as reference for the next scene. This helps in keeping:

* same character
* same background
* same objects

### Feedback and Correction

After generating a scene, it is evaluated. If something goes wrong (like identity mismatch), only that scene is regenerated instead of the whole video.

---

## System Flow

Long Prompt
→ Scene Segmentation
→ Dependency Detection
→ Scene Generation
→ Consistency Check
→ Selective Correction
→ Final Video

---

## Project Structure

```
project/
│── README.md
│── requirements.txt
│── main.py

├── configs/
│   ├── settings.yaml
│   ├── generation/
│   │   ├── backend.yaml
│   │   ├── prompt_builder.yaml
│   │   └── continuity.yaml
│   └── prompts/
│       ├── seg_system.txt
│       ├── seg_user.txt
│       ├── dep_system.txt
│       ├── dep_user.txt
│       ├── verify_system.txt
│       ├── verify_user.txt
│       ├── json_repair_system.txt
│       ├── json_repair_user.txt
│       └── style_guide.txt

├── src/
│   ├── text/
│   │   ├── segmentation.py
│   │   └── dependency.py
│   ├── llm/
│   │   ├── prompts.py
│   │   ├── client.py
│   │   └── parsing.py
│   ├── models/
│   │   └── diffusion/
│   │       ├── cogvideox_backend.py
│   │       └── svd_backend.py
│   ├── generation/
│   │   ├── backend_router.py
│   │   └── scene_generator.py
│   ├── continuity/
│   │   ├── keyframe_selector.py
│   │   └── consistency_scorer.py
│   └── evaluation/
│       └── metrics.py

├── scripts/
│   ├── 00_create_dummy_videos.py
│   └── 01_extract_clips.py

├── data/
│   └── prompts/

├── evaluation/
│   └── metrics.yaml

└── outputs/
    └── runs/
```

---

## How to Run

```bash
git clone <your-repo-link>
cd project
pip install -r requirements.txt
python main.py
```

---

## Evaluation

The generated videos are evaluated based on:

* how well they match the text
* whether the same character is maintained
* whether scenes are consistent with each other

---

## Author

Rohan Pol
M.Tech (AI & ML)
