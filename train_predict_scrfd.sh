#!/usr/bin/env bash
set -eo pipefail

for scrfd_type in "2.5" "10" "34"; do
  for cls in "all" "bird" "cat" "cat_like" "dog" "dog_like" "horse_like" "small_animals"; do
    PYTHONPATH='/mnt/data/afarec/code/face_detection/SCRFD/':$PYTHONPATH \
    python /mnt/data/afarec/code/face_detection/SCRFD/tools/train.py \
      "$(dirname "$0")/configs/scrfd_${scrfd_type}/scrfd_${scrfd_type}g_${cls}.py" \
      --seed 0 \
      --work-dir "./work_dir/scrfd_${scrfd_type}_${cls}/"

    PYTHONPATH='/mnt/data/afarec/code/face_detection/SCRFD/':$PYTHONPATH \
    python /mnt/data/afarec/code/face_detection/SCRFD/tools/test_widerface.py \
      "$(dirname "$0")/configs/scrfd_${scrfd_type}/scrfd_${scrfd_type}g_${cls}.py" \
      "./work_dir/scrfd_${scrfd_type}_${cls}/weights/lastest.pth" \
      --mode 2 \
      --save-preds \
      --out "./work_dir/scrfd_${scrfd_type}_${cls}/"
  done

  PYTHONPATH='/mnt/data/afarec/code/face_detection/SCRFD/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/SCRFD/tools/test_widerface.py \
    "$(dirname "$0")/configs/scrfd_${scrfd_type}/scrfd_${scrfd_type}g_all.py" \
    "$(dirname "$0")/weights/model_pretrained_${scrfd_type}GF.pth" \
    --mode 2 \
    --save-preds \
    --out "./work_dir/scrfd_${scrfd_type}_pretrained/"
done
