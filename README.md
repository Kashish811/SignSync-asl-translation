---
title: SignSync
sdk: docker
app_port: 7860
pinned: false
---

# SignSync: sign language video to English text

SignSync translates continuous American Sign Language (ASL) video into English. It reads the pose of the signer, builds four skeleton graphs (body, left hand, right hand, face), encodes them with a spatio-temporal graph neural network and decodes English with an mT5 language model. A web app lets you upload a clip, browse honest examples from the official test set, and see how well the system performs.

## What is in this repository

| Part | What it is |
|---|---|
| `app/main.py`, `app/engine.py` | FastAPI backend: examples, statistics and live translation |
| `app/static/index.html` | The web frontend (single page, light and dark mode, mobile friendly) |
| `scripts/make_examples.py` | Builds the example cards and result numbers from real evaluation files |
| `scripts/fetch_model.py` | Downloads the model code and weights (they are not stored here) |
| `Dockerfile` | Deployment on a Hugging Face Space |

The evaluation pipeline (data preparation, test-set evaluation, shuffled-input and part-zeroing controls, error analysis) and the presentation notebook live in the companion `SignSync_PC` project.

## Credits and scope

The model architecture, the pretrained weights (`how2sign_pose_only_slt.pth`) and the pose-extraction demo code are from **Uni-Sign** (Li et al., [github.com/ZechengLi19/Uni-Sign](https://github.com/ZechengLi19/Uni-Sign)), released under CC BY-NC 4.0: **non-commercial use only, attribution required**. They are downloaded at setup time and are not redistributed in this repository. Data: [How2Sign](https://how2sign.github.io/) (Duarte et al., CVPR 2021). Language model: mT5-base (Google).

What this project adds: preparing the official How2Sign data, a reproducible evaluation on the official test set, controls that test whether the model uses the poses, error and duplicate analysis, and this application.

## Results (official How2Sign test set)

Numbers come from `results/main_test_metrics.json` of the evaluation run and are shown live on the site. BLEU-4 is about 13.75 and ROUGE-L about 30.7 on the 2,349 of 2,357 test clips that have a pose file. About 15% of sentences are exact. Many outputs are fluent but wrong and some repeat a phrase. See the Limits section of the site.

## Run it

**On the college PC (GPU), using the model already set up:**
```bash
# 1. build the example cards and numbers from your real results
cd ~/Downloads/SignSync_PC
.venv/bin/python ~/Downloads/SignSync_App/scripts/make_examples.py --results results --out ~/Downloads/SignSync_App/app/data
# 2. start the app
cd ~/Downloads/SignSync_App && bash run_local.sh      # open http://localhost:8000
```

**Anywhere else:**
```bash
pip install -r requirements.txt            # examples and statistics only
pip install -r requirements-model.txt      # adds live translation
python scripts/fetch_model.py              # downloads ~3.5 GB of model assets
python -m uvicorn main:app --app-dir app --port 8000
```

## Deploy (Hugging Face Space)

Create a Docker Space, push this repository to it, and the `Dockerfile` does the rest. Live translation on the free CPU tier is slow and untested; the examples and results pages work either way.

## Privacy

Uploaded videos are written to a temporary file, processed and deleted immediately. They are not stored or shared.

## Limits

Research system, not an interpreter. ASL only, one sentence per clip, trained on frontal studio recordings. Do not use for medical, legal or safety-critical communication.
