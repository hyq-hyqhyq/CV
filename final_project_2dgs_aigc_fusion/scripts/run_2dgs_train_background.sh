#!/usr/bin/env bash
# Train a Mip-NeRF 360 background scene with an external 2DGS repository.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_2dgs_train_background.sh [options]

Options:
  --data_dir          Mip-NeRF 360 scene path. Default: data/mipnerf360/counter
  --output_dir        Output directory. Default: outputs/background/2dgs
  --iterations        Training iterations. Default: 7000
  --resolution        Lower is heavier; 4 is recommended as a safe starting point. Default: 4
  --external_2dgs     External 2DGS repo path. Default: configs/project_paths.yaml external_2dgs
  --config            Path config YAML. Default: configs/project_paths.yaml
  --                  Extra arguments appended to external train.py.
  -h, --help          Show this help message.
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

DATA_DIR=""
OUTPUT_DIR=""
ITERATIONS="7000"
RESOLUTION="4"
EXTERNAL_2DGS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    --external_2dgs)
      EXTERNAL_2DGS="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
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
EXTERNAL_2DGS="${EXTERNAL_2DGS:-$(yaml_value external_2dgs)}"
DATA_DIR="${DATA_DIR:-data/mipnerf360/counter}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-outputs}/background/2dgs}"

DATA_DIR="$(resolve_path "$DATA_DIR")"
OUTPUT_DIR="$(resolve_path "$OUTPUT_DIR")"
EXTERNAL_2DGS="$(resolve_path "$EXTERNAL_2DGS")"
TRAIN_PY="$EXTERNAL_2DGS/train.py"
LOG_FILE="$OUTPUT_DIR/train.log"

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "Error: 2DGS train.py not found: $TRAIN_PY" >&2
  echo "Clone or add your 2DGS repository under external/ and update configs/project_paths.yaml." >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: background data directory not found: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "2DGS repository: $EXTERNAL_2DGS"
echo "Background data: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Iterations: $ITERATIONS"
echo "Resolution/downscale: $RESOLUTION"
echo "Log file: $LOG_FILE"

python "$TRAIN_PY" \
  -s "$DATA_DIR" \
  -m "$OUTPUT_DIR" \
  --iterations "$ITERATIONS" \
  -r "$RESOLUTION" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

