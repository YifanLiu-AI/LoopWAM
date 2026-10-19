#!/usr/bin/env bash
# Link RoboCasa365 and robosuite assets into this checkout.
#
# Preferred usage from LoopWAM root:
#
#   ROBOCASA_ASSETS_SRC=/path/to/robocasa_assets \
#   ROBOSUITE_ASSETS_SRC=/path/to/robosuite_assets \
#     bash eval_robocasa.sh prepare
#
# Backward-compatible variable names are also accepted:
#   ASSET_SRC_ROOT       == ROBOCASA_ASSETS_SRC
#   ROBOSUITE_ASSET_SRC  == ROBOSUITE_ASSETS_SRC
#
# IMPORTANT:
#   ROBOCASA_ASSETS_SRC must point to an external RoboCasa asset root that
#   contains these large subdirectories:
#     fixtures/  objects/  textures/  generative_textures/
#
#   ROBOSUITE_ASSETS_SRC must point to the robosuite models/assets root that
#   contains at least:
#     robots/panda/robot.xml
#
# The script intentionally creates one link per large RoboCasa subdirectory so
# repository-bundled RoboCasa asset folders such as arenas/, box_links/, scenes/
# remain visible in robocasa/robocasa/models/assets.
#
# It creates:
#   robocasa/robocasa/models/assets/fixtures             -> $ROBOCASA_ASSETS_SRC/fixtures
#   robocasa/robocasa/models/assets/objects              -> $ROBOCASA_ASSETS_SRC/objects
#   robocasa/robocasa/models/assets/textures             -> $ROBOCASA_ASSETS_SRC/textures
#   robocasa/robocasa/models/assets/generative_textures  -> $ROBOCASA_ASSETS_SRC/generative_textures
#   robosuite/robosuite/models/assets                    -> $ROBOSUITE_ASSETS_SRC
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOCASA_ASSETS_SRC="${ROBOCASA_ASSETS_SRC:-${ASSET_SRC_ROOT:-}}"
ROBOSUITE_ASSETS_SRC="${ROBOSUITE_ASSETS_SRC:-${ROBOSUITE_ASSET_SRC:-}}"

ROBOCASA_ASSETS_DST="${ROOT}/robocasa/robocasa/models/assets"
ROBOSUITE_ASSETS_DST="${ROOT}/robosuite/robosuite/models/assets"

fail() {
  echo "[setup_symlinks] ERROR: $*" >&2
  exit 1
}

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "${path}" ]] || fail "missing ${label}: ${path}"
}

if [[ -z "${ROBOCASA_ASSETS_SRC}" ]]; then
  fail "set ROBOCASA_ASSETS_SRC=/path/to/robocasa_assets (or legacy ASSET_SRC_ROOT)."
fi
if [[ -z "${ROBOSUITE_ASSETS_SRC}" ]]; then
  fail "set ROBOSUITE_ASSETS_SRC=/path/to/robosuite_assets (or legacy ROBOSUITE_ASSET_SRC)."
fi

require_dir "${ROBOCASA_ASSETS_SRC}" "RoboCasa assets root"
require_dir "${ROBOSUITE_ASSETS_SRC}" "robosuite assets root"

mkdir -p "${ROBOCASA_ASSETS_DST}"
for name in fixtures objects textures generative_textures; do
  src="${ROBOCASA_ASSETS_SRC}/${name}"
  dst="${ROBOCASA_ASSETS_DST}/${name}"
  require_dir "${src}" "RoboCasa asset subdirectory ${name}"
  rm -rf "${dst}"
  ln -sfn "$(realpath "${src}")" "${dst}"
  echo "[setup_symlinks] ${dst} -> $(realpath "${src}")"
done

if [[ ! -f "${ROBOSUITE_ASSETS_SRC}/robots/panda/robot.xml" ]]; then
  fail "invalid ROBOSUITE_ASSETS_SRC=${ROBOSUITE_ASSETS_SRC}; expected ${ROBOSUITE_ASSETS_SRC}/robots/panda/robot.xml. This must be robosuite/robosuite/models/assets, not RoboCasa assets."
fi

mkdir -p "$(dirname "${ROBOSUITE_ASSETS_DST}")"
rm -rf "${ROBOSUITE_ASSETS_DST}"
ln -sfn "$(realpath "${ROBOSUITE_ASSETS_SRC}")" "${ROBOSUITE_ASSETS_DST}"
echo "[setup_symlinks] ${ROBOSUITE_ASSETS_DST} -> $(realpath "${ROBOSUITE_ASSETS_SRC}")"

echo "[setup_symlinks] done"
