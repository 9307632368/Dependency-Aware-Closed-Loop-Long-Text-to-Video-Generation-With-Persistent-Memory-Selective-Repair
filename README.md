# folder structure 

LongText2Video/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── configs/
│   ├── settings.yaml
│   ├── prompts/
│   │   ├── seg_system.txt
│   │   ├── seg_user.txt
│   │   ├── verify_system.txt
│   │   ├── verify_user.txt
│   │   ├── dep_system.txt
│   │   ├── dep_user.txt
│   │   ├── json_repair_system.txt
│   │   ├── json_repair_user.txt
│   │   └── style_guide.txt
│   ├── generation/
│   │   ├── backend.yaml
│   │   ├── prompt_builder.yaml
│   │   └── continuity.yaml
│   └── evaluation/
│       └── metrics.yaml
│
├── data/
│   ├── prompts/
│   ├── metadata/
│   │   ├── scenes/
│   │   ├── dependencies/
│   │   └── scene_packets/
│   ├── references/
│   │   ├── keyframes/
│   │   ├── characters/
│   │   └── locations/
│   ├── video_raw/
│   ├── video_clips/
│   └── cache/
│
├── outputs/
│   ├── runs/
│   │   └── run_YYYYMMDD_HHMM/
│   │       ├── logs/
│   │       ├── raw_llm/
│   │       ├── scenes.json
│   │       ├── dependencies.json
│   │       ├── scene_packets.json
│   │       ├── memory_state.json
│   │       ├── selected_references.json
│   │       ├── prompts_used/
│   │       ├── frames/
│   │       ├── clips/
│   │       ├── stitched_video/
│   │       └── metrics.json
│   └── figures/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── prompts.py
│   │   ├── parsing.py
│   │   └── repair.py
│   │
│   ├── text/
│   │   ├── __init__.py
│   │   ├── segmentation.py
│   │   ├── dependency.py
│   │   ├── sentence_utils.py
│   │   ├── postprocess.py
│   │   ├── scene_packet.py
│   │   └── packet_builder.py
│   │
│   ├── continuity/
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   ├── reference_bank.py
│   │   ├── manager.py
│   │   ├── drift.py
│   │   └── state_tracker.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py
│   │   ├── scene_generator.py
│   │   └── backend/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── svd_backend.py
│   │       └── cogvideox_backend.py
│   │
│   ├── video/
│   │   ├── __init__.py
│   │   ├── frames.py
│   │   ├── stitch.py
│   │   ├── transitions.py
│   │   └── io.py
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── continuity_metrics.py
│   │   ├── story_metrics.py
│   │   ├── ablation_runner.py
│   │   └── sanity.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── run_text_pipeline.py
│   │   ├── run_generation_pipeline.py
│   │   └── run_full_pipeline.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── io.py
│       ├── seed.py
│       └── paths.py
│
└── scripts/
    ├── __init__.py
    ├── run_single_prompt.py
    ├── run_batch_prompts.py
    ├── run_scene_generation.py
    ├── run_long_video.py
    ├── extract_clips.py
    └── make_report_figs.py

..    