#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR='/mnt/data/afarec/code/face_detection/SCRFD/'

for scrfd_type in "2.5" "10" "34"; do
  for cls in "all" "bird" "cat" "cat_like" "dog" "dog_like" "horse_like" "small_animals"; do
    PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
    python "${ROOT_DIR}tools/train.py" \
      "${ROOT_DIR}configs/scrfd_${scrfd_type}/scrfd_${scrfd_type}g_${cls}.py" \
      --seed 0 \
      --no-validate \
      --work-dir "${ROOT_DIR}work_dir/scrfd_${scrfd_type}_${cls}/"

    PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
    python "${ROOT_DIR}tools/test_widerface.py" \
      "${ROOT_DIR}configs/scrfd_${scrfd_type}/scrfd_${scrfd_type}g_${cls}.py" \
      "${ROOT_DIR}work_dir/scrfd_${scrfd_type}_${cls}/latest.pth" \
      --mode 2 \
      --save-preds \
      --out "${ROOT_DIR}work_dir/scrfd_${scrfd_type}_${cls}/"
  done

  PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
  python "${ROOT_DIR}tools/test_widerface.py" \
    "${ROOT_DIR}configs/scrfd_${scrfd_type}/scrfd_${scrfd_type}g_all.py" \
    "${ROOT_DIR}weights/model_pretrained_${scrfd_type}GF.pth" \
    --mode 2 \
    --save-preds \
    --out "${ROOT_DIR}work_dir/scrfd_${scrfd_type}_pretrained/"
done
