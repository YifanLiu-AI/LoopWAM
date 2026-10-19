#!/usr/bin/env python
"""Create a LeRobot dataset with one pre-stitched RoboTwin video stream.

The source dataset must contain the three RoboTwin camera videos:

* observation.images.cam_high
* observation.images.cam_left_wrist
* observation.images.cam_right_wrist

The destination dataset keeps metadata/tabular files, drops the three video
features from ``meta/info.json``, and adds one ``observation.images.robotwin``
video feature. Videos are laid out as the training dataloader does online:
top camera 256x320 above two wrist cameras resized to 128x160 each.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


SRC_VIDEO_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
DST_VIDEO_KEY = "observation.images.robotwin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--dst", type=Path, required=True, help="Destination LeRobot dataset root.")
    parser.add_argument("--workers", type=int, default=4, help="Number of ffmpeg jobs to run in parallel.")
    parser.add_argument("--crf", type=int, default=16, help="H.264 CRF. Lower is higher quality/larger files.")
    parser.add_argument("--preset", default="medium", help="ffmpeg libx264 preset.")
    parser.add_argument(
        "--gop",
        type=int,
        default=2,
        help="Fixed keyframe interval in frames. Default 2 matches the small RoboTwin dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recreate output videos that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing files.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N episodes, for testing.")
    return parser.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH.")


def copy_non_video_tree(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] copy metadata/tabular files: {src} -> {dst}")
        return

    def ignore_large_dirs(_dir: str, names: list[str]) -> set[str]:
        ignored = {"videos"} if "videos" in names else set()
        ignored.update(name for name in names if name.startswith("text_embeds_cache"))
        return ignored

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=ignore_large_dirs, dirs_exist_ok=True)


def update_info_json(src: Path, dst: Path, dry_run: bool) -> None:
    info_path = dst / "meta" / "info.json"
    if dry_run:
        print(f"[dry-run] update {info_path}: single video feature {DST_VIDEO_KEY} -> [384, 320, 3]")
        return

    with (src / "meta" / "info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)

    features = dict(info["features"])
    missing = [key for key in SRC_VIDEO_KEYS if key not in features]
    if missing:
        raise KeyError(f"Missing source video features in info.json: {missing}")

    first_video = features[SRC_VIDEO_KEYS[0]]
    for key in SRC_VIDEO_KEYS:
        features.pop(key, None)
    features[DST_VIDEO_KEY] = {
        "dtype": "video",
        "shape": [384, 320, 3],
        "names": first_video.get("names", ["height", "width", "rgb"]),
        "info": {
            "video.height": 384,
            "video.width": 320,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": info["fps"],
            "video.channels": 3,
            "has_audio": False,
        },
    }
    info["features"] = features
    info["total_videos"] = int(info["total_episodes"])

    tmp_path = info_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        f.write("\n")
    os.replace(tmp_path, info_path)


def episode_indices(src: Path) -> list[int]:
    indices = []
    with (src / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            indices.append(int(obj["episode_index"]))
    return indices


def video_path(root: Path, ep: int, video_key: str, chunks_size: int) -> Path:
    return root / f"videos/chunk-{ep // chunks_size:03d}/{video_key}/episode_{ep:06d}.mp4"


def stitch_one(
    src: Path,
    dst: Path,
    ep: int,
    chunks_size: int,
    crf: int,
    preset: str,
    gop: int,
    overwrite: bool,
) -> tuple[Path, str]:
    out_path = video_path(dst, ep, DST_VIDEO_KEY, chunks_size)
    if out_path.exists() and not overwrite:
        return out_path, "skipped"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    inputs = [video_path(src, ep, key, chunks_size) for key in SRC_VIDEO_KEYS]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source videos for episode {ep}: {missing}")

    filter_complex = (
        "[0:v]scale=320:256:flags=lanczos[top];"
        "[1:v]scale=160:128:flags=lanczos[left];"
        "[2:v]scale=160:128:flags=lanczos[right];"
        "[left][right]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[outv]"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for path in inputs:
        cmd.extend(["-i", str(path)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-an",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ]
    )
    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return out_path, "written"


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    if src == dst:
        raise ValueError("--src and --dst must be different directories.")
    if args.gop < 1:
        raise ValueError(f"--gop must be >= 1, got {args.gop}")
    if not (src / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {src / 'meta' / 'info.json'}")
    if not args.dry_run:
        require_ffmpeg()

    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    chunks_size = int(info["chunks_size"])
    episodes = episode_indices(src)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise RuntimeError("No episodes found.")

    print(f"[stitch] src={src}")
    print(f"[stitch] dst={dst}")
    print(f"[stitch] episodes={len(episodes)} output_key={DST_VIDEO_KEY}")
    print(f"[stitch] encoder=libx264 preset={args.preset} crf={args.crf} gop={args.gop}")

    copy_non_video_tree(src, dst, args.dry_run)
    update_info_json(src, dst, args.dry_run)

    if args.dry_run:
        ep = episodes[0]
        print(f"[dry-run] stitch episode {ep:06d} -> {video_path(dst, ep, DST_VIDEO_KEY, chunks_size)}")
        return

    written = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                stitch_one,
                src,
                dst,
                ep,
                chunks_size,
                args.crf,
                args.preset,
                args.gop,
                args.overwrite,
            )
            for ep in episodes
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Stitching videos"):
            _path, status = future.result()
            if status == "written":
                written += 1
            else:
                skipped += 1

    print(f"[stitch] done: written={written} skipped={skipped} dst={dst}")


if __name__ == "__main__":
    main()
