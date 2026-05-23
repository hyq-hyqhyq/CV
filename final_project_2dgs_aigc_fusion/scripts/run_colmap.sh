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

colmap feature_extractor \
  --database_path "$DATABASE_PATH" \
  --image_path "$IMAGE_DIR" \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 1

case "$MATCHER" in
  exhaustive)
    colmap exhaustive_matcher \
      --database_path "$DATABASE_PATH" \
      --SiftMatching.use_gpu 1
    ;;
  sequential)
    colmap sequential_matcher \
      --database_path "$DATABASE_PATH" \
      --SiftMatching.use_gpu 1
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

