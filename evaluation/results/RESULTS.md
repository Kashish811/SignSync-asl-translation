# Measured results (this project)

| run | split | clips | control/dropped parts | B1 | B2 | B3 | B4 | ROUGE-L | BLEURT |
|---|---|---|---|---|---|---|---|---|---|
| main_test | test | 2349 | none/- | 38.51 | 25.50 | 18.35 | 13.75 | 30.66 | nan |
| smoke | test | 20 | none/- | 39.80 | 28.28 | 20.64 | 15.07 | 28.30 | nan |
| sub_drop_body | test | 500 | none/body | 25.43 | 13.24 | 8.13 | 5.30 | 17.24 | nan |
| sub_drop_face_all | test | 500 | none/face_all | 32.75 | 19.96 | 13.65 | 9.69 | 23.96 | nan |
| sub_drop_left | test | 500 | none/left | 36.72 | 23.37 | 16.32 | 11.72 | 27.52 | nan |
| sub_drop_right | test | 500 | none/right | 19.62 | 8.42 | 4.32 | 2.46 | 12.68 | nan |
| sub_full | test | 500 | none/- | 39.26 | 26.05 | 18.72 | 13.89 | 30.88 | nan |
| sub_hands_off | test | 500 | none/left,right | 13.91 | 4.53 | 1.92 | 0.77 | 7.68 | nan |
| sub_shuffled | test | 500 | shuffle/- | 15.15 | 4.49 | 1.76 | 0.66 | 7.09 | nan |

## Published How2Sign results (BLEU-4; to verify against each paper before citing)
| method | BLEU-4 | note |
|---|---|---|
| H2S-only baselines | ~0.9-2.3 | no large-scale pretraining |
| Tarres et al. | 8.03 | |
| SignFormer-GCN | 8.53 | ST-GCN keypoints + RGB features |
| YouTube-ASL | 12.39 | pretrain + finetune |
| Uni-Sign (RGB-pose) | 14.9 | ST-GCN pose encoder + mT5 |
| Uni-Sign (pose-only) | ~14.5 | read from table; verify |
| SSVP-SLT | 15.5 | |
