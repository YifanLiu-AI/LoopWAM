# RoboCasa365 (eval client)

Self-contained RoboCasa365 **simulator client** used by LoopWAM evaluation. Large mesh/texture assets are soft-linked, not stored in git.

## Layout

```text
third_party/RoboCasa365/
├── robocasa365_inference_client_pro.sh   # parallel client launcher
├── eval_robocasa_client_pro.py           # HTTP client + env rollout
├── setup_symlinks.sh                     # link fixtures / objects / textures / robosuite assets
├── robocasa/                             # RoboCasa365 code
└── robosuite/                            # robosuite code
```

The LoopWAM **policy server** stays in `experiments/robocasa/serve_robocasa_policy.py`. Launch everything from the repo root with `eval_robocasa.sh`.

## One-time asset links

```bash
  ROBOCASA_ASSETS_SRC=/path/to/robocasa_assets \
  ROBOSUITE_ASSETS_SRC=/path/to/robosuite_assets \
  bash eval_robocasa.sh prepare
```

## Run (via repo root)

```bash
# 1) Download released RoboCasa365 weights into checkpoints/robocasa365/
bash scripts/download_robocasa365.sh

# 2) Start the policy server
conda activate loopwam
bash eval_robocasa.sh serve

# 3) Start the policy client
conda activate robocasa365
bash eval_robocasa.sh client \
  --server_url http://127.0.0.1:7891 \
  --task_groups "RecycleBottlesByType,WaffleReheat;\
ArrangeBreadBasket,WeighIngredients;\
BreadSelection,CuttingToolSelection;\
GarnishPancake,ArrangeTea" \
  --split pretrain \
  --replan_steps 48 \
  --num_trials 50 \
  --save_video_every 0 \
  --run_name robocasa365_eval_unseen1
```

Videos default to `third_party/RoboCasa365/eval_videos/`. Logs default to `third_party/RoboCasa365/logs/eval/`.
