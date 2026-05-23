#!/usr/bin/env bash
# Export or stage a mesh/point-cloud representation from external 2DGS outputs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_2dgs_mesh.sh --asset object_a --checkpoint_dir outputs/object_a/2dgs --out_dir outputs/object_a/mesh

Options:
  --asset             Asset name, for example object_a or background.
  --checkpoint_dir    2DGS training output directory.
  --out_dir           Mesh output directory.
  --external_2dgs     External 2DGS repo path. Default: configs/project_paths.yaml external_2dgs
  --export_command    Optional command template to run. Tokens are replaced:
                      {checkpoint_dir}, {out_dir}, {asset}
  --config            Path config YAML. Default: configs/project_paths.yaml
  -h, --help          Show this help message.

TODO:
  2DGS mesh extraction commands differ across forks. Prefer the official mesh
  extraction, TSDF fusion, or export command from your selected 2DGS repository.
  If no direct command exists, use an exported point cloud/PLY as an intermediate
  and convert/clean it manually in Blender or MeshLab.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$PROJECT_ROOT/configs/project_paths.yaml"

yaml_value() {
  local key="$1"
  python3 - "$CONFIG" "$key" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.is_file():
    print("")
    raise SystemExit
for line in path.read_text(encoding="utf-8").splitlines():
    clean = line.split("#", 1)[0].strip()
    if not clean or ":" not in clean:
        continue
    k, v = clean.split(":", 1)
    if k.strip() == key:
        print(v.strip().strip('"').strip("'"))
        break
PY
}

resolve_path() {
  local path_text="$1"
  if [[ "$path_text" = /* ]]; then
    printf '%s\n' "$path_text"
  else
    printf '%s\n' "$PROJECT_ROOT/$path_text"
  fi
}

ASSET=""
CHECKPOINT_DIR=""
OUT_DIR=""
EXTERNAL_2DGS=""
EXPORT_COMMAND=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset)
      ASSET="$2"
      shift 2
      ;;
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --external_2dgs)
      EXTERNAL_2DGS="$2"
      shift 2
      ;;
    --export_command)
      EXPORT_COMMAND="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$ASSET" || -z "$CHECKPOINT_DIR" || -z "$OUT_DIR" ]]; then
  echo "Error: --asset, --checkpoint_dir and --out_dir are required." >&2
  usage
  exit 1
fi

OUTPUT_ROOT="$(yaml_value output_root)"
EXTERNAL_2DGS="${EXTERNAL_2DGS:-$(yaml_value external_2dgs)}"
CHECKPOINT_DIR="$(resolve_path "$CHECKPOINT_DIR")"
OUT_DIR="$(resolve_path "$OUT_DIR")"
EXTERNAL_2DGS="$(resolve_path "$EXTERNAL_2DGS")"

mkdir -p "$PROJECT_ROOT/${OUTPUT_ROOT:-outputs}/object_a/mesh" \
  "$PROJECT_ROOT/${OUTPUT_ROOT:-outputs}/background/mesh" \
  "$OUT_DIR"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Error: checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

echo "2DGS repository: $EXTERNAL_2DGS"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Mesh output directory: $OUT_DIR"

if [[ -n "$EXPORT_COMMAND" ]]; then
  CMD="${EXPORT_COMMAND//\{checkpoint_dir\}/$CHECKPOINT_DIR}"
  CMD="${CMD//\{out_dir\}/$OUT_DIR}"
  CMD="${CMD//\{asset\}/$ASSET}"
  echo "Running user-provided export command:"
  echo "$CMD"
  bash -lc "$CMD"
  exit 0
fi

PLY_CANDIDATE="$(find "$CHECKPOINT_DIR" -type f -name '*.ply' | sort | tail -n 1 || true)"
if [[ -n "$PLY_CANDIDATE" ]]; then
  FALLBACK_OUT="$OUT_DIR/${ASSET}_intermediate.ply"
  cp "$PLY_CANDIDATE" "$FALLBACK_OUT"
  echo "No explicit export command was provided."
  echo "Fallback: copied latest PLY as an intermediate representation:"
  echo "  $FALLBACK_OUT"
  echo "TODO: convert or clean this PLY into a textured mesh using your selected 2DGS fork, Blender, MeshLab, or TSDF fusion."
  exit 0
fi

echo "Error: no mesh export command was provided and no PLY file was found under $CHECKPOINT_DIR." >&2
echo "TODO: check your 2DGS repository for mesh extraction / TSDF / export commands, then rerun with --export_command." >&2
exit 1

