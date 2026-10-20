"""RoboCasa365 rollout collector -> LeRobot v2.1 (self-shooting / self-distillation).

This module records policy rollouts produced during RoboCasa evaluation and
writes them as a **byte-compatible LeRobot v2.1 dataset**, identical in layout
to the original RoboCasa365 expert data under
``.../<Task>/<date>/lerobot``. The resulting directories can be dropped straight
into ``configs/data/robocasa365_mixed.yaml`` as extra ``dataset_dirs`` and read
by the project's vendored v2.1 loader with ZERO changes.

Why a hand-rolled writer (not ``lerobot`` 0.3.3's high-level API)?
  * The training dataloader is a *vendored* lerobot pinned to
    ``CODEBASE_VERSION = "v2.1"`` (one parquet per episode, ``chunk-{:03d}``,
    ``meta/episodes.jsonl`` + ``meta/episodes_stats.jsonl``).
  * lerobot 0.3.3 writes v3.0 (multi-episode chunks, different meta), which the
    v2.1 loader cannot read. So we mirror the v2.1 layout directly with
    ``pyarrow`` / ``imageio`` / ``json`` — no lerobot dependency at all.

What matches the original data exactly (verified against a real sample's
``meta/info.json`` + parquet schema):
  * parquet columns & dtypes:
      annotation.human.task_description : int64            (task_index into tasks.jsonl)
      annotation.human.task_name        : int64
      observation.state                 : fixed_size_list<double>[16]   (NATIVE 16D)
      action                            : fixed_size_list<double>[12]   (NATIVE 12D)
      next.reward                       : float32
      next.done                         : bool
      timestamp                         : float32
      frame_index                       : int64
      episode_index                     : int64
      index                             : int64
      task_index                        : int64
  * 3 videos per episode, h264 / yuv420p / fps=20, 256x256, keys
    ``observation.images.{robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand}``.
  * meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}.

Extra vs. the expert data (both are additive and harmless to the loader):
  * ``meta/episodes.jsonl`` records a ``"success"`` bool per episode. The
    project's dataset reads this directly (RobotVideoDataset._get_sample_episode_success),
    so NO separate ``add_success_to_rollout_meta.py`` step is needed.
  * ``meta/rollout_info.json`` records provenance (checkpoint, seeds, counts).

Dimension handling (store NATIVE, exactly like expert data — the training
``RobocasaSliceTransform`` slices 16->14 / 12->11 in the dataloader):
  * state : the client already builds a COMPACT 14D state
    (``get_state_from_obs``); we scatter it back to the raw 16D layout, filling
    the dropped constant-zero base-quaternion dims [3,4] with 0.
  * action: the model returns a COMPACT 11D action; we expand it to the native
    12D env layout (the client's own ``_compact_action_to_native`` convention:
    native[4]=control_mode, native[3]=const 0).

Usage — see ``rollout_collector_integration_example.py`` for the exact hooks to
add to ``eval_robocasa_client_old.py``. In short::

    collector = RolloutCollector(
        out_root="data/robocasa365_rollout",
        fps=20,
        video_size=(256, 256),
    )
    ep = collector.start_episode(task_name="CloseFridge",
                                 task_description="Close the fridge.")
    # each control step, BEFORE env.step, with the raw model outputs:
    ep.add_step(images_client_order=images,          # 3x PIL/np, client order
                state_compact_14d=state,             # np [14]
                action_compact_11d=raw_action,       # np [11] (executed)
                reward=float(reward), done=bool(done))
    collector.finish_episode(ep, success=bool(done))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - hard dependency
    raise ImportError(
        "rollout_collector requires pyarrow (present in the lerobot/robocasa env)."
    ) from exc

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - fall back to imageio v1 API
    import imageio  # type: ignore

from PIL import Image


# ---------------------------------------------------------------------------
# Fixed conventions (must match training config + original data)
# ---------------------------------------------------------------------------

CODEBASE_VERSION = "v2.1"
ROBOT_TYPE = "PandaOmron"
CHUNKS_SIZE = 1000  # original info.json: chunks_size=1000 -> single chunk-000 in practice

# Camera order as the client sends it (eval_robocasa_client_old.get_images_from_obs).
CLIENT_CAMERA_ORDER: Tuple[str, str, str] = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)

# LeRobot video feature keys (info.json "features"), in the SAME order the
# original data declares them (eye_in_hand first). Order only affects
# total_videos accounting / feature dict order; per-key files are independent.
VIDEO_KEYS: Tuple[str, str, str] = (
    "robot0_eye_in_hand",
    "robot0_agentview_left",
    "robot0_agentview_right",
)

STATE_RAW_DIM = 16
ACTION_RAW_DIM = 12
# RobocasaSliceTransform.STATE_KEEP: 14 compact dims -> positions in the raw 16D.
STATE_KEEP: Tuple[int, ...] = (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def _scatter_state_14_to_16(state14: np.ndarray) -> np.ndarray:
    """Compact 14D PandaOmron state -> native 16D (dropped base-quat dims = 0)."""
    state14 = np.asarray(state14, dtype=np.float64).reshape(-1)
    if state14.shape[0] == STATE_RAW_DIM:
        return state14  # already native
    if state14.shape[0] != len(STATE_KEEP):
        raise ValueError(
            f"Expected compact 14D or native 16D state, got {state14.shape[0]}D."
        )
    raw16 = np.zeros(STATE_RAW_DIM, dtype=np.float64)
    raw16[list(STATE_KEEP)] = state14
    return raw16


def _expand_action_11_to_12(action11: np.ndarray) -> np.ndarray:
    """Compact 11D model action -> native 12D env action.

    Mirrors eval_robocasa_client_old._compact_action_to_native:
        native[0:3]  = a[0:3]     (base_motion xyz)
        native[3]    = 0          (base_motion 4th dim, constant 0)
        native[4]    = a[3]       (control_mode)
        native[5:8]  = a[4:7]     (eef position delta)
        native[8:11] = a[7:10]    (eef rotation delta)
        native[11]   = a[10]      (gripper_close)
    """
    action = np.asarray(action11, dtype=np.float64).reshape(-1)
    if action.shape[0] == ACTION_RAW_DIM:
        return action  # already native
    if action.shape[0] != 11:
        raise ValueError(
            f"Expected compact 11D or native 12D action, got {action.shape[0]}D."
        )
    native = np.zeros(ACTION_RAW_DIM, dtype=np.float64)
    native[0:3] = action[0:3]
    native[4] = action[3]
    native[5:8] = action[4:7]
    native[8:11] = action[7:10]
    native[11] = action[10]
    return native


def _to_uint8_rgb(img: Union[np.ndarray, "Image.Image"], size_hw: Tuple[int, int]) -> np.ndarray:
    """Coerce a camera frame to a contiguous uint8 RGB array of shape (H, W, 3)."""
    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")
    h, w = size_hw
    if pil.size != (w, h):
        pil = pil.resize((w, h), resample=Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Per-episode buffer
# ---------------------------------------------------------------------------

@dataclass
class EpisodeBuffer:
    task_name: str
    task_description: str
    # frames[cam_key] -> list of (H, W, 3) uint8
    frames: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    states16: List[np.ndarray] = field(default_factory=list)
    actions12: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        for key in VIDEO_KEYS:
            self.frames[key] = []

    def add_step(
        self,
        images_client_order: Sequence[Union[np.ndarray, "Image.Image"]],
        state_compact_14d: np.ndarray,
        action_compact_11d: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        video_size: Tuple[int, int] = (256, 256),
    ) -> None:
        if len(images_client_order) != 3:
            raise ValueError(f"Expected 3 camera images, got {len(images_client_order)}.")
        # Map from client order to per-video-key storage.
        by_name = dict(zip(CLIENT_CAMERA_ORDER, images_client_order))
        for key in VIDEO_KEYS:
            self.frames[key].append(_to_uint8_rgb(by_name[key], video_size))
        self.states16.append(_scatter_state_14_to_16(state_compact_14d))
        self.actions12.append(_expand_action_11_to_12(action_compact_11d))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.states16)


# ---------------------------------------------------------------------------
# Stats (episodes_stats.jsonl) — format required by the vendored v2.1 loader
# ---------------------------------------------------------------------------

def _feature_stats_vector(array: np.ndarray) -> Dict[str, list]:
    """min/max/mean/std/count for a 2D array (T, D); keepdims over axis 0 -> (D,)."""
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def _feature_stats_scalar(array: np.ndarray) -> Dict[str, list]:
    """Stats for a per-frame scalar feature, kept as shape (1,) like the loader expects."""
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    return {
        "min": [float(np.min(array))],
        "max": [float(np.max(array))],
        "mean": [float(np.mean(array))],
        "std": [float(np.std(array))],
        "count": [int(array.shape[0])],
    }


def _feature_stats_image(frames: List[np.ndarray]) -> Dict[str, list]:
    """Per-channel image stats normalized to [0,1], shape (3,1,1) — matches
    compute_stats.compute_episode_stats for image/video features.

    We sub-sample frames the same spirit as lerobot (bounded cost), reduce over
    (frame, H, W) keeping channel, then divide by 255.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("empty frame list for image stats")
    # bounded sub-sampling
    max_samples = min(n, 1000)
    idx = np.round(np.linspace(0, n - 1, max_samples)).astype(int)
    stack = np.stack([frames[i] for i in idx], axis=0).astype(np.float64)  # (S,H,W,3)
    stack = stack.transpose(0, 3, 1, 2)  # (S,3,H,W)
    axes = (0, 2, 3)
    out = {
        "min": np.min(stack, axis=axes, keepdims=True),
        "max": np.max(stack, axis=axes, keepdims=True),
        "mean": np.mean(stack, axis=axes, keepdims=True),
        "std": np.std(stack, axis=axes, keepdims=True),
    }
    # squeeze the leading batch dim -> (3,1,1); normalize to [0,1]
    result = {k: np.squeeze(v / 255.0, axis=0).tolist() for k, v in out.items()}
    result["count"] = [int(n)]
    return result


# ---------------------------------------------------------------------------
# Collector: one LeRobot v2.1 dataset per task (matching original layout)
# ---------------------------------------------------------------------------

class _TaskWriter:
    """Owns one ``<out_root>/<task>/lerobot`` v2.1 dataset, appending episodes.

    Resumable: on construction it reads any existing meta so repeated eval runs
    append rather than overwrite (episode_index / global frame index continue).
    """

    def __init__(
        self,
        root: Path,
        task_name: str,
        fps: int,
        video_size: Tuple[int, int],
        video_codec: str = "libx264",
        video_pix_fmt: str = "yuv420p",
    ) -> None:
        self.root = root
        self.task_name = task_name
        self.fps = int(fps)
        self.video_h, self.video_w = int(video_size[0]), int(video_size[1])
        self.video_codec = video_codec
        self.video_pix_fmt = video_pix_fmt

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for key in VIDEO_KEYS:
            (self.root / "videos" / "chunk-000" / f"observation.images.{key}").mkdir(
                parents=True, exist_ok=True
            )

        # tasks.jsonl: natural-language descriptions + the task_name literal, each
        # mapped to a stable task_index. Mirrors the original convention where the
        # per-frame `annotation.human.task_description` stores the description index
        # and `annotation.human.task_name` / `task_index` store the task_name index.
        self._task_to_index: Dict[str, int] = {}
        self._episodes: List[dict] = []
        self._episodes_stats: List[dict] = []
        self._total_frames = 0

        self._load_existing_meta()

    # -- resume support ----------------------------------------------------
    def _load_existing_meta(self) -> None:
        tasks_path = self.root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            for line_no, line in enumerate(tasks_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._task_to_index[rec["task"]] = int(rec["task_index"])

        episodes_path = self.root / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            for line_no, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._episodes.append(rec)
                self._total_frames += int(rec["length"])

        stats_path = self.root / "meta" / "episodes_stats.jsonl"
        if stats_path.exists():
            for line_no, line in enumerate(stats_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    self._episodes_stats.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    @property
    def next_episode_index(self) -> int:
        return len(self._episodes)

    def _task_index(self, task: str) -> int:
        if task not in self._task_to_index:
            self._task_to_index[task] = len(self._task_to_index)
        return self._task_to_index[task]

    # -- write one episode -------------------------------------------------
    def add_episode(self, ep: EpisodeBuffer, success: bool, env_seed: Optional[int]) -> int:
        length = len(ep)
        if length == 0:
            raise ValueError("refusing to write an empty episode")

        episode_index = self.next_episode_index

        desc_index = self._task_index(ep.task_description)
        name_index = self._task_index(ep.task_name)

        # ---- parquet columns (native 16D state / 12D action) -------------
        states = np.stack(ep.states16, axis=0).astype(np.float64)   # (T,16)
        actions = np.stack(ep.actions12, axis=0).astype(np.float64)  # (T,12)
        rewards = np.asarray(ep.rewards, dtype=np.float32)           # (T,)
        dones = np.asarray(ep.dones, dtype=bool)                     # (T,)
        timestamps = (np.arange(length, dtype=np.float32) / self.fps)
        frame_index = np.arange(length, dtype=np.int64)
        episode_col = np.full((length,), episode_index, dtype=np.int64)
        global_index = np.arange(self._total_frames, self._total_frames + length, dtype=np.int64)
        # IMPORTANT: match the original RoboCasa365 convention. The dataloader sets
        # the training instruction via ``item["task"] = tasks[task_index]``; in the
        # expert data ``task_index`` points to the NATURAL-LANGUAGE description
        # (e.g. index 0 -> "Increase the toaster oven temperature."), NOT the bare
        # task name. So the per-frame task_index must equal desc_index.
        task_index_col = np.full((length,), desc_index, dtype=np.int64)
        desc_col = np.full((length,), desc_index, dtype=np.int64)
        name_col = np.full((length,), name_index, dtype=np.int64)

        table = pa.table(
            {
                "annotation.human.task_description": pa.array(desc_col, type=pa.int64()),
                "annotation.human.task_name": pa.array(name_col, type=pa.int64()),
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(states.reshape(-1), type=pa.float64()), STATE_RAW_DIM
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(actions.reshape(-1), type=pa.float64()), ACTION_RAW_DIM
                ),
                "next.reward": pa.array(rewards, type=pa.float32()),
                "next.done": pa.array(dones, type=pa.bool_()),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(frame_index, type=pa.int64()),
                "episode_index": pa.array(episode_col, type=pa.int64()),
                "index": pa.array(global_index, type=pa.int64()),
                "task_index": pa.array(task_index_col, type=pa.int64()),
            }
        )
        data_path = self.root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        pq.write_table(table, str(data_path))

        # ---- videos (one mp4 per camera key) -----------------------------
        for key in VIDEO_KEYS:
            frames = ep.frames[key]
            if len(frames) != length:
                raise ValueError(
                    f"camera {key} has {len(frames)} frames but episode length is {length}"
                )
            video_path = (
                self.root
                / "videos"
                / "chunk-000"
                / f"observation.images.{key}"
                / f"episode_{episode_index:06d}.mp4"
            )
            self._write_mp4(video_path, frames)

        # ---- per-episode stats ------------------------------------------
        stats: Dict[str, Any] = {}
        for key in VIDEO_KEYS:
            stats[f"observation.images.{key}"] = _feature_stats_image(ep.frames[key])
        stats["annotation.human.task_description"] = _feature_stats_scalar(desc_col)
        stats["annotation.human.task_name"] = _feature_stats_scalar(name_col)
        stats["observation.state"] = _feature_stats_vector(states)
        stats["action"] = _feature_stats_vector(actions)
        stats["next.reward"] = _feature_stats_scalar(rewards.astype(np.float64))
        stats["next.done"] = _feature_stats_scalar(dones.astype(np.float64))
        stats["timestamp"] = _feature_stats_scalar(timestamps.astype(np.float64))
        stats["frame_index"] = _feature_stats_scalar(frame_index.astype(np.float64))
        stats["episode_index"] = _feature_stats_scalar(episode_col.astype(np.float64))
        stats["index"] = _feature_stats_scalar(global_index.astype(np.float64))
        stats["task_index"] = _feature_stats_scalar(task_index_col.astype(np.float64))

        self._episodes_stats.append({"episode_index": episode_index, "stats": stats})

        # ---- episode record (with success!) ------------------------------
        self._episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [ep.task_description],
                "length": length,
                "success": bool(success),
                "env_seed": (int(env_seed) if env_seed is not None else None),
            }
        )
        self._total_frames += length

        # flush meta after every episode so a crash mid-run stays consistent.
        self._flush_meta()
        return episode_index

    def _write_mp4(self, path: Path, frames: List[np.ndarray]) -> None:
        # h264 / yuv420p / fps to match original video_info.
        writer = imageio.get_writer(
            str(path),
            format="FFMPEG",
            mode="I",
            fps=self.fps,
            codec=self.video_codec,
            pixelformat=self.video_pix_fmt,
            macro_block_size=1,  # allow 256x256 without silent resizing
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()

    # -- meta files --------------------------------------------------------
    def _video_feature(self) -> dict:
        return {
            "dtype": "video",
            "shape": [self.video_h, self.video_w, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(self.fps),
                "video.codec": "h264",
                "video.pix_fmt": self.video_pix_fmt,
                "video.is_depth_map": False,
                "has_audio": False,
            },
            "info": {
                "video.height": self.video_h,
                "video.width": self.video_w,
                "video.codec": "h264",
                "video.pix_fmt": self.video_pix_fmt,
                "video.is_depth_map": False,
                "video.fps": float(self.fps),
                "video.channels": 3,
                "has_audio": False,
            },
        }

    def _build_info(self) -> dict:
        features: Dict[str, Any] = {}
        for key in VIDEO_KEYS:
            features[f"observation.images.{key}"] = self._video_feature()
        features["annotation.human.task_description"] = {"dtype": "int64", "shape": [1]}
        features["annotation.human.task_name"] = {"dtype": "int64", "shape": [1]}
        features["observation.state"] = {"dtype": "float64", "shape": [STATE_RAW_DIM]}
        features["action"] = {"dtype": "float64", "shape": [ACTION_RAW_DIM]}
        features["next.reward"] = {"dtype": "float32", "shape": [1]}
        features["next.done"] = {"dtype": "bool", "shape": [1]}
        features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
        features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

        total_episodes = len(self._episodes)
        total_chunks = max(1, (max(total_episodes - 1, 0) // CHUNKS_SIZE) + 1)
        return {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": ROBOT_TYPE,
            "total_episodes": total_episodes,
            "total_frames": int(self._total_frames),
            "total_tasks": len(self._task_to_index),
            "total_videos": total_episodes * len(VIDEO_KEYS),
            "total_chunks": total_chunks,
            "chunks_size": CHUNKS_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }

    def _flush_meta(self) -> None:
        meta = self.root / "meta"

        _atomic_write_json(meta / "info.json", self._build_info())

        # tasks.jsonl (sorted by index for determinism)
        tasks_sorted = sorted(self._task_to_index.items(), key=lambda kv: kv[1])
        _atomic_write_jsonl(
            meta / "tasks.jsonl",
            [{"task_index": idx, "task": task} for task, idx in tasks_sorted],
        )

        _atomic_write_jsonl(meta / "episodes.jsonl", self._episodes)
        _atomic_write_jsonl(meta / "episodes_stats.jsonl", self._episodes_stats)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def _atomic_write_json_if_absent(path: Path, obj: Any) -> bool:
    """Write JSON only if the target file does not already exist.

    We intentionally use direct writes instead of tempfile+os.replace because
    some shared/TOS filesystems allow normal writes but reject replace/rename.
    Returns True if we created the file, False if it already existed.
    """
    if path.exists():
        return False
    try:
        _atomic_write_json(path, obj)
        return True
    except PermissionError:
        if path.exists():
            return False
        raise


def _atomic_write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class RolloutCollector:
    """Top-level entry point. Manages one ``_TaskWriter`` per task name.

    Directory layout produced (one LeRobot v2.1 dataset per task)::

        <out_root>/<TaskName>/lerobot/
            meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, rollout_info.json}
            data/chunk-000/episode_000000.parquet ...
            videos/chunk-000/observation.images.<cam>/episode_000000.mp4 ...
    """

    def __init__(
        self,
        out_root: Union[str, Path],
        fps: int = 20,
        video_size: Tuple[int, int] = (256, 256),
        subdir: str = "lerobot",
        video_codec: str = "libx264",
        video_pix_fmt: str = "yuv420p",
        provenance: Optional[dict] = None,
    ) -> None:
        self.out_root = Path(out_root)
        self.fps = int(fps)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.subdir = subdir
        self.video_codec = video_codec
        self.video_pix_fmt = video_pix_fmt
        self.provenance = provenance or {}
        self._writers: Dict[str, _TaskWriter] = {}

    def _writer(self, task_name: str) -> _TaskWriter:
        if task_name not in self._writers:
            root = self.out_root / task_name / self.subdir
            writer = _TaskWriter(
                root=root,
                task_name=task_name,
                fps=self.fps,
                video_size=self.video_size,
                video_codec=self.video_codec,
                video_pix_fmt=self.video_pix_fmt,
            )
            if self.provenance:
                rollout_info_path = root / "meta" / "rollout_info.json"
                _atomic_write_json_if_absent(rollout_info_path, self.provenance)
            self._writers[task_name] = writer
        return self._writers[task_name]

    def start_episode(self, task_name: str, task_description: str) -> EpisodeBuffer:
        return EpisodeBuffer(task_name=task_name, task_description=task_description)

    def finish_episode(
        self,
        ep: EpisodeBuffer,
        success: bool,
        env_seed: Optional[int] = None,
        min_length: int = 1,
    ) -> Optional[int]:
        """Persist ``ep`` as one episode. Returns its episode_index, or None if
        skipped (too short). Success and failure are BOTH stored by design —
        failed rollouts still supervise the video branch downstream.
        """
        if len(ep) < max(1, int(min_length)):
            return None
        writer = self._writer(ep.task_name)
        return writer.add_episode(ep, success=success, env_seed=env_seed)

    def episode_count(self, task_name: str) -> int:
        return self._writer(task_name).next_episode_index
