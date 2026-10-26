# SignSync – college Ubuntu PC (VS Code terminal, no Colab)
Open this folder in VS Code (File ▸ Open Folder), open a terminal (Ctrl+`), then run the scripts in order.
If a script fails, copy the LAST 30 lines of output to Claude; do not edit code.

## Before you start (one-time, ~5 min)
1. Get the 3 CSVs onto the PC: from Google Drive `SignSync/How2Sign/` download
   how2sign_realigned_train.csv, how2sign_realigned_val.csv, how2sign_realigned_test.csv into `~/Downloads/How2Sign/` (or edit CSV_DIR in config.env).
2. (Only for the demo) download 2-3 clips from Drive `How2Sign/test_rgb_front_clips/`.
3. Check `df -h .` -> at least 40 GB free in this folder's disk.

## Run
bash 00_setup.sh        # env + clone + deps; prints nvidia-smi and torch/CUDA status           (CPU, ~5 min)
bash 01_download.sh     # mT5 + How2Sign poses + checkpoint (~12 GB)                              (network, ~10-40 min)
bash 02_prepare.sh      # labels + unzip + check; all 3 splits must say PASS                      (CPU, ~10 min)
bash 03_smoke.sh        # 20 test clips on the GPU  -> note sec_per_clip                          (GPU, minutes)
bash 04_eval_full.sh    # FULL test set: headline BLEU/ROUGE                                      (GPU: ≈ 2,300 clips × sec_per_clip)
bash 05_gnn_controls.sh # shuffled input + drop each GNN part, 500-clip subset -> results/RESULTS.md (GPU)
bash 06_demo.sh clip.mp4

GPU time guard: steps 00-02 need no GPU. After 03 compute: full_test_minutes = 2300 × sec_per_clip / 60. Run 05 only if it fits.
Outputs: results/*_metrics.json, results/*_predictions.csv (reference vs prediction + sentence BLEU), results/RESULTS.md.
Report only numbers from results/. Cite Uni-Sign (Li et al.), How2Sign (Duarte et al., CVPR 2021), mT5, ST-GCN.
What is borrowed vs yours: see VIVA_NOTES.md.
