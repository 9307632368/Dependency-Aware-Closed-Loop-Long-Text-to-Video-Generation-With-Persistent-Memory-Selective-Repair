# Dependency-Aware Closed-Loop Long Text-to-Video Generation  Persistent Visual Memory and Selective Scene Repair



## Overview

This project focuses on long text-to-video generation, where a long input prompt is divided into multiple scenes and generated step by step. The main aim is to reduce common problems in long video generation such as identity drift, location mismatch, object inconsistency, and broken scene continuity.

Instead of generating the whole video at once, this project follows a structured pipeline. It first performs scene segmentation, then checks dependencies between scenes, generates video scene-wise, stores continuity information, and applies repair when inconsistencies are detected.

## Main Idea

Most text-to-video models work better for short clips. When the prompt becomes long and contains multiple connected scenes, the generated video may lose consistency. For example, the same character may look different in later scenes, or the location may change even when it should remain the same.

This project handles that problem using dependency-aware scene generation, memory-based continuity, and selective repair.

## System Flow

```text
Long Text Prompt
        ↓
Scene Segmentation
        ↓
Dependency Detection
        ↓
Scene Packet Creation
        ↓
Scene-wise Video Generation
        ↓
Continuity Tracking
        ↓
Consistency Evaluation
        ↓
Selective Scene Repair
        ↓
Final Video Output
```

## Key Components

### Scene Segmentation

The long prompt is divided into smaller scene-level descriptions. This makes the generation process easier to control and evaluate.

### Dependency Detection

The system checks whether a scene depends on previous scenes. For example, a scene may depend on the same character, same location, same object, or previous event.

### Scene-wise Generation

Each scene is generated separately using video generation backends such as CogVideoX, SVD, dummy backend, or hybrid backend depending on the configuration.

### Continuity Memory

Important information from previous scenes is stored and reused. This includes character appearance, locations, props, style, and selected keyframes.

### Consistency Scoring

Generated scenes are checked for continuity problems such as identity drift, object loss, location mismatch, and story inconsistency.

### Selective Repair

If a scene has a problem, the repair module decides whether to regenerate the scene, improve the prompt, change references, or apply a repair policy.

## Project Structure

```text
project/
│── README.md
│── requirements.txt
│── implementation workflow.txt
│
├── configs/
│   ├── settings.yaml
│   │
│   ├── diffusion/
│   │   ├── model.yaml
│   │   └── sampling.yaml
│   │
│   ├── evaluation/
│   │   └── metrics.yaml
│   │
│   ├── generation/
│   │   ├── backend.yaml
│   │   ├── continuity.yaml
│   │   └── prompt_builder.yaml
│   │
│   └── prompts/
│       ├── dep_system.txt
│       ├── dep_user.txt
│       ├── json_repair_system.txt
│       ├── json_repair_user.txt
│       ├── repair_system.txt
│       ├── repair_user.txt
│       ├── seg_system.txt
│       ├── seg_user.txt
│       ├── style_guide.txt
│       ├── verify_system.txt
│       └── verify_user.txt
│
├── data/
│   ├── metadata/
│   │   ├── dependencies/
│   │   ├── scenes/
│   │   └── scene_packets/
│   │
│   ├── prompts/
│   │   └── prompt_001.txt
│   │
│   ├── references/
│   │   ├── characters/
│   │   ├── keyframes/
│   │   └── locations/
│   │
│   ├── video_clips/
│   └── video_raw/
│
├── outputs/
│   ├── figures/
│   ├── generated/
│   │   ├── cogvideox/
│   │   └── svd/
│   │
│   └── runs/
│
├── scripts/
│   ├── extract_clips.py
│   ├── make_report_figs.py
│   ├── run_batch_prompts.py
│   ├── run_long_video.py
│   ├── run_scene_generation.py
│   ├── run_single_prompt.py
│   └── __init__.py
│
└── src/
    ├── main.py
    ├── __init__.py
    │
    ├── continuity/
    │   ├── consistency_scorer.py
    │   ├── constraint_builder.py
    │   ├── drift.py
    │   ├── extract.py
    │   ├── keyframe_selector.py
    │   ├── manager.py
    │   ├── memory.py
    │   ├── package.py
    │   ├── reference_bank.py
    │   ├── state_tracker.py
    │   ├── story_schema.py
    │   └── __init__.py
    │
    ├── diffusion/
    │   ├── conditioning.py
    │   ├── model.py
    │   ├── sampler.py
    │   └── __init__.py
    │
    ├── eval/
    │   ├── ablation_runner.py
    │   ├── continuity_metrics.py
    │   ├── metrics.py
    │   ├── sanity.py
    │   ├── story_metrics.py
    │   └── __init__.py
    │
    ├── generation/
    │   ├── prompt_builder.py
    │   ├── retry.py
    │   ├── scene_generator.py
    │   ├── __init__.py
    │   │
    │   └── backend/
    │       ├── base.py
    │       ├── cogvideox_backend.py
    │       ├── common.py
    │       ├── dummy_backend.py
    │       ├── factory.py
    │       ├── hybrid_backend.py
    │       ├── router.py
    │       ├── svd_backend.py
    │       └── __init__.py
    │
    ├── llm/
    │   ├── client.py
    │   ├── parsing.py
    │   ├── prompts.py
    │   ├── repair.py
    │   └── __init__.py
    │
    ├── pipeline/
    │   ├── run_full_pipeline.py
    │   ├── run_generation_pipeline.py
    │   ├── run_text_pipeline.py
    │   └── __init__.py
    │
    ├── repair/
    │   ├── failure_classifier.py
    │   ├── repair_policy.py
    │   ├── scene_repair.py
    │   └── __init__.py
    │
    ├── text/
    │   ├── dependency.py
    │   ├── packet_builder.py
    │   ├── postprocess.py
    │   ├── scene_packet.py
    │   ├── segmentation.py
    │   ├── sentence_splitter.py
    │   ├── sentence_utils.py
    │   └── __init__.py
    │
    ├── utils/
    │   ├── io.py
    │   ├── logger.py
    │   ├── paths.py
    │   ├── seed.py
    │   └── __init__.py
    │
    └── video/
        ├── frames.py
        ├── io.py
        ├── stitch.py
        ├── transitions.py
        ├── vae.py
        └── __init__.py
```

## How to Run

```bash
pip install -r requirements.txt
python -m src.main
```

or run specific scripts:

```bash
python scripts/run_single_prompt.py
python scripts/run_long_video.py
python scripts/run_scene_generation.py
```

## Notes

The project is organized around scene-level generation, continuity memory, evaluation, and repair. The codebase separates text processing, generation, diffusion-related modules, continuity handling, repair logic, and video utilities so that each part can be improved independently.

## some samples
Click the image below to watch the stitched generated video.

[![Watch Final Video](outputs/runs/run_long_20260318_220725/generation/scene_001/attempt_00/frames/frame_004.png)](outputs/runs/run_long_20260318_220725/stitched_video/stitched_video.mp4)

### Scene-wise Preview

| Scene | Preview | Video |
|---|---|---|
| Scene 01 | ![Scene 01](outputs/runs/run_long_20260318_220725/generation/scene_001/attempt_00/frames/frame_004.png) | [Watch Scene 01](outputs/runs/run_long_20260318_220725/generation/scene_001/attempt_00/scene_001.mp4) |
| Scene 02 | ![Scene 02](outputs/runs/run_long_20260318_220725/generation/scene_002/attempt_00/frames/frame_004.png) | [Watch Scene 02](outputs/runs/run_long_20260318_220725/generation/scene_002/attempt_00/scene_002.mp4) |
| Scene 03 | ![Scene 03](outputs/runs/run_long_20260318_220725/generation/scene_003/attempt_00/frames/frame_004.png) | [Watch Scene 03](outputs/runs/run_long_20260318_220725/generation/scene_003/attempt_00/scene_003.mp4) |
| Scene 04 | ![Scene 04](outputs/runs/run_long_20260318_220725/generation/scene_004/attempt_00/frames/frame_004.png) | [Watch Scene 04](outputs/runs/run_long_20260318_220725/generation/scene_004/attempt_00/scene_004.mp4) |


## Authors

Rohan Pol  
M.Tech AI & ML