# Minecraft Mob Detection: FCOS vs YOLO

## Results
| Model | mAP | mAP@0.5 |
|-------|-----|---------|
| FCOS | 0.2530 | 0.4210 |
| YOLOv8s | 0.5012 | 0.7659 |

## Summary
YOLOv8s shows better detection quality compared to FCOS on Minecraft mobs dataset.

## Project Structure
- `configs/fcos/fcos_minecraft.py` - FCOS configuration
- `artifacts/` - models, metrics, inference results, videos
- `minecraft_project.ipynb` - training and evaluation pipeline

## Artifacts
- FCOS weights: `artifacts/fcos/best_coco_bbox_mAP_epoch_10.pth`
- YOLO weights: `artifacts/yolo/exp_aug/weights/best.pt`
- Inference videos: `artifacts/videos/`
- Test images with detections: `artifacts/inference/`
