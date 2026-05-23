#!/usr/bin/env bash
# Run external Magic123 for single-image-to-3D object C generation.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_magic123_object_c.sh --input_png data/object_c/foreground_png/object_c.png --run_command '<command template>'

Options:
  --input_png       Foreground PNG input with alpha.
  --output_dir      Magic123 output directory. Default: outputs/object_c/magic123
  --mesh_out        Expected mesh export path. Default: outputs/object_c/mesh/object_c.obj
  --external_repo   External Magic123 repo path. Default: configs/project_paths.yaml external_magic123
  --run_command     Command template to execute inside external Magic123 repo.
                    Tokens are replaced: {input_png}, {output_dir}, {mesh_out}
  --config          Path config YAML. Default: configs/project_paths.yaml
  -h, --help        Show this help message.

Notes:
  Magic123 dependencies are heavy. Run on Ubuntu + CUDA and keep pretrained
  weights outside this repository. Different forks expose different commands,
  so provide --run_command according to the installed version.
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

INPUT_PNG=""
OUTPUT_DIR=""
MESH_OUT=""
EXTERNAL_REPO=""
RUN_COMMAND=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_png)
      INPUT_PNG="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --mesh_out)
      MESH_OUT="$2"
      shift 2
      ;;
    --external_repo)
      EXTERNAL_REPO="$2"
      shift 2
      ;;
    --run_command)
      RUN_COMMAND="$2"
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

OUTPUT_ROOT="$(yaml_value output_root)"
EXTERNAL_REPO="${EXTERNAL_REPO:-$(yaml_value external_magic123)}"
INPUT_PNG="${INPUT_PNG:-data/object_c/foreground_png/object_c.png}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-outputs}/object_c/magic123}"
MESH_OUT="${MESH_OUT:-${OUTPUT_ROOT:-outputs}/object_c/mesh/object_c.obj}"

INPUT_PNG="$(resolve_path "$INPUT_PNG")"
OUTPUT_DIR="$(resolve_path "$OUTPUT_DIR")"
MESH_OUT="$(resolve_path "$MESH_OUT")"
EXTERNAL_REPO="$(resolve_path "$EXTERNAL_REPO")"

if [[ ! -f "$INPUT_PNG" ]]; then
  echo "Error: foreground PNG not found: $INPUT_PNG" >&2
  exit 1
fi
if [[ ! -d "$EXTERNAL_REPO" ]]; then
  echo "Error: Magic123 repository not found: $EXTERNAL_REPO" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$(dirname "$MESH_OUT")"

if [[ -z "$RUN_COMMAND" || "$RUN_COMMAND" == TODO* ]]; then
  echo "Error: --run_command is required because Magic123 command lines vary across forks." >&2
  echo "Example template:"
  echo "  --run_command 'python launch.py --config configs/magic123.yaml data.image_path={input_png} save_dir={output_dir}'"
  echo "After generation, export or copy the mesh to: $MESH_OUT"
  exit 1
fi

CMD="${RUN_COMMAND//\{input_png\}/$INPUT_PNG}"
CMD="${CMD//\{output_dir\}/$OUTPUT_DIR}"
CMD="${CMD//\{mesh_out\}/$MESH_OUT}"

echo "Magic123 repository: $EXTERNAL_REPO"
echo "Input foreground PNG: $INPUT_PNG"
echo "Output directory: $OUTPUT_DIR"
echo "Expected mesh path: $MESH_OUT"
echo "Running command:"
echo "$CMD"

cd "$EXTERNAL_REPO"
bash -lc "$CMD"

echo "Magic123 command finished."
if [[ ! -f "$MESH_OUT" ]]; then
  echo "Warning: expected mesh was not found at $MESH_OUT."
  echo "Export or copy the final mesh there before running normalization and Blender fusion."
fi
