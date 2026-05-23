#!/usr/bin/env bash
# Generate object B with an external threestudio SDS text-to-3D pipeline.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_threestudio_object_b.sh --config_name <valid-threestudio-config> [options]

Options:
  --prompt             Text prompt for object B.
  --output_dir         Output directory. Default: outputs/object_b/threestudio
  --config_name        threestudio config name/path. Choose one from your installed version.
  --external_repo      External threestudio repo path. Default: configs/project_paths.yaml external_threestudio
  --config             Path config YAML. Default: configs/project_paths.yaml
  --                  Extra arguments appended to launch.py.
  -h, --help           Show this help message.

Notes:
  SDS configs vary by threestudio version. Select a valid DreamFusion, SJC,
  ProlificDreamer, or other SDS config from external/threestudio/configs.
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

PROMPT="a small stylized ceramic robot toy, glossy white body, blue circular eyes, simple rounded shape, clean texture, product photography style"
OUTPUT_DIR=""
CONFIG_NAME=""
EXTERNAL_REPO=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --config_name)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --external_repo)
      EXTERNAL_REPO="$2"
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
EXTERNAL_REPO="${EXTERNAL_REPO:-$(yaml_value external_threestudio)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-outputs}/object_b/threestudio}"

EXTERNAL_REPO="$(resolve_path "$EXTERNAL_REPO")"
OUTPUT_DIR="$(resolve_path "$OUTPUT_DIR")"
LAUNCH_PY="$EXTERNAL_REPO/launch.py"
LOG_FILE="$OUTPUT_DIR/train.log"

if [[ -z "$CONFIG_NAME" || "$CONFIG_NAME" == TODO* ]]; then
  echo "Error: --config_name must be a real threestudio SDS config from your installation." >&2
  echo "Open external/threestudio/configs and choose a DreamFusion/SJC/ProlificDreamer-style config." >&2
  exit 1
fi
if [[ ! -f "$LAUNCH_PY" ]]; then
  echo "Error: threestudio launch.py not found: $LAUNCH_PY" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/${OUTPUT_ROOT:-outputs}/object_b/mesh"

echo "threestudio repository: $EXTERNAL_REPO"
echo "Config name/path: $CONFIG_NAME"
echo "Prompt: $PROMPT"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

cd "$EXTERNAL_REPO"
python "$LAUNCH_PY" \
  --config "$CONFIG_NAME" \
  --train \
  system.prompt_processor.prompt="$PROMPT" \
  trial_dir="$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "Training finished or stopped by the external command."
echo "TODO: export the generated mesh with the command supported by this threestudio version."
echo "Expected final mesh path: $PROJECT_ROOT/${OUTPUT_ROOT:-outputs}/object_b/mesh/object_b.obj or object_b.glb"

