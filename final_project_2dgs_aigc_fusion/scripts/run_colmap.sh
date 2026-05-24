#!/usr/bin/env bash
# Run the standard COLMAP sparse reconstruction and undistortion pipeline.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_colmap.sh --image_dir data/object_a/colmap/images --output_dir data/object_a/colmap

Arguments:
  --image_dir      Directory containing input images.
  --output_dir     COLMAP workspace directory.
  --matcher        exhaustive or sequential. Default: exhaustive.
  -h, --help       Show this help message.
EOF
}

IMAGE_DIR=""
OUTPUT_DIR=""
MATCHER="exhaustive"

help_has_option() {
  local command_name="$1"
  local option_name="$2"
  colmap "$command_name" -h 2>&1 | grep -q -- "$option_name"
}

append_if_supported() {
  local array_name="$1"
  local command_name="$2"
  local option_name="$3"
  local option_value="$4"
  if help_has_option "$command_name" "$option_name"; then
    eval "$array_name+=(\"$option_name\" \"$option_value\")"
    return 0
  fi
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image_dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --matcher)
      MATCHER="$2"
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

if [[ -z "$IMAGE_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Error: --image_dir and --output_dir are required." >&2
  usage
  exit 1
fi

if ! command -v colmap >/dev/null 2>&1; then
  echo "Error: colmap executable not found. Install COLMAP and make sure it is on PATH." >&2
  exit 1
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Error: image directory not found: $IMAGE_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/sparse" "$OUTPUT_DIR/dense"
DATABASE_PATH="$OUTPUT_DIR/database.db"

echo "COLMAP image directory: $IMAGE_DIR"
echo "COLMAP output directory: $OUTPUT_DIR"
echo "COLMAP database: $DATABASE_PATH"

FEATURE_ARGS=(
  --database_path "$DATABASE_PATH"
  --image_path "$IMAGE_DIR"
)
append_if_supported FEATURE_ARGS feature_extractor "--ImageReader.single_camera" 1 || true
if append_if_supported FEATURE_ARGS feature_extractor "--SiftExtraction.use_gpu" 1; then
  echo "Using COLMAP feature GPU option: --SiftExtraction.use_gpu 1"
elif append_if_supported FEATURE_ARGS feature_extractor "--FeatureExtraction.use_gpu" 1; then
  echo "Using COLMAP feature GPU option: --FeatureExtraction.use_gpu 1"
else
  echo "Warning: no recognized feature extraction GPU option found; using COLMAP defaults."
fi

colmap feature_extractor "${FEATURE_ARGS[@]}"

case "$MATCHER" in
  exhaustive)
    MATCH_ARGS=(--database_path "$DATABASE_PATH")
    if append_if_supported MATCH_ARGS exhaustive_matcher "--SiftMatching.use_gpu" 1; then
      echo "Using COLMAP matching GPU option: --SiftMatching.use_gpu 1"
    elif append_if_supported MATCH_ARGS exhaustive_matcher "--FeatureMatching.use_gpu" 1; then
      echo "Using COLMAP matching GPU option: --FeatureMatching.use_gpu 1"
    else
      echo "Warning: no recognized feature matching GPU option found; using COLMAP defaults."
    fi
    colmap exhaustive_matcher "${MATCH_ARGS[@]}"
    ;;
  sequential)
    MATCH_ARGS=(--database_path "$DATABASE_PATH")
    if append_if_supported MATCH_ARGS sequential_matcher "--SiftMatching.use_gpu" 1; then
      echo "Using COLMAP matching GPU option: --SiftMatching.use_gpu 1"
    elif append_if_supported MATCH_ARGS sequential_matcher "--FeatureMatching.use_gpu" 1; then
      echo "Using COLMAP matching GPU option: --FeatureMatching.use_gpu 1"
    else
      echo "Warning: no recognized feature matching GPU option found; using COLMAP defaults."
    fi
    colmap sequential_matcher "${MATCH_ARGS[@]}"
    ;;
  *)
    echo "Error: --matcher must be exhaustive or sequential." >&2
    exit 1
    ;;
esac

colmap mapper \
  --database_path "$DATABASE_PATH" \
  --image_path "$IMAGE_DIR" \
  --output_path "$OUTPUT_DIR/sparse"

MODEL_DIR="$OUTPUT_DIR/sparse/0"
if [[ ! -d "$MODEL_DIR" ]]; then
  MODEL_DIR="$(find "$OUTPUT_DIR/sparse" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1 || true)"
fi

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
  echo "Error: COLMAP mapper did not produce a sparse model under $OUTPUT_DIR/sparse." >&2
  exit 1
fi

colmap image_undistorter \
  --image_path "$IMAGE_DIR" \
  --input_path "$MODEL_DIR" \
  --output_path "$OUTPUT_DIR/dense" \
  --output_type COLMAP

echo "COLMAP finished."
echo "Sparse model: $MODEL_DIR"
echo "Undistorted dense dataset: $OUTPUT_DIR/dense"
