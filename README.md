<div align="center">

# SignSync

### Continuous ASL-to-English translation with a part-aware graph neural network

Sign language video in. English text out. Built on body-pose keypoints only.

[![Hosted results](https://img.shields.io/badge/Hosted%20results-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Kashish811/SignSync-asl-translation)
[![Colab demo](https://img.shields.io/badge/Live%20demo-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](docs/SignSync_Colab_Demo.ipynb)
![BLEU-4](https://img.shields.io/badge/BLEU--4-13.75-2563eb?style=for-the-badge)
![ROUGE-L](https://img.shields.io/badge/ROUGE--L-30.7-7c3aed?style=for-the-badge)
![Licence](https://img.shields.io/badge/Use-non--commercial-64748b?style=for-the-badge)

</div>

---

## Overview

SignSync translates continuous American Sign Language (ASL) video into English. It reads **poses, not pixels**: RTMPose extracts body, hand and face keypoints, a spatial-temporal graph convolutional network (ST-GCN) encodes four skeleton graphs, and an mT5-base decoder writes the sentence.

| | |
| --- | --- |
| **Input** | ASL video (133 whole-body keypoints per frame) |
| **Graph encoder** | Four part graphs with ST-GCN encoders (4.6 M parameters) |
| **Decoder** | mT5-base (582 M parameters), four-beam search |
| **Data** | How2Sign, official test split |
| **Interface** | FastAPI backend and a web frontend |

> **Attribution.** The architecture, trained weights and pose pipeline are **Uni-Sign's** ([Li et al., 2025](https://arxiv.org/abs/2501.15187)), pose-only configuration, used under CC BY-NC 4.0 (non-commercial, attribution required). **No model was trained in this project.** The data is [How2Sign](https://how2sign.github.io) (Duarte et al., CVPR 2021).
> This repository adds an independent evaluation, control experiments, analysis scripts, an executed notebook that traces tensors through every stage, and the web application.

---

## Results

Evaluated on **2,349 of the 2,357** How2Sign test clips (eight had no pose file).

<div align="center">

| Metric | Value |
| :--- | :---: |
| **BLEU-4** (sacreBLEU) | **13.75** (95% bootstrap CI 13.04 to 14.47) |
| **ROUGE-L** (own implementation) | **30.7** |

</div>

For reference, the Uni-Sign paper reports BLEU-4 14.5 for the same configuration. The 0.75 gap was not investigated.

### What does the trained model depend on?

The trained model was rerun on 500 test clips with parts of the input removed at test time. This measures dependence. It is not a retraining ablation.

| Input | BLEU-4 |
| :--- | :---: |
| Full input | 13.89 |
| Left hand removed | 11.72 |
| Whole face removed | 9.69 |
| Body removed | 5.30 |
| **Right hand removed** | **2.46** |
| Both hands removed | 0.77 |
| Frames shuffled | 0.66 |

The model leans far more on the right hand than the left, and shuffling the frames destroys the output, so the score comes from pose content and its temporal order.

**Known limits:** between 22% and 29% of outputs end in a repeated phrase. Results come from one dataset and one signing domain. BLEU and ROUGE measure word overlap, not meaning.

---

## How it works

```
Video  ->  RTMPose (133 keypoints/frame)
       ->  4 part graphs: body 9 | left hand 21 | right hand 21 | face 18
       ->  ST-GCN encoders (spatial + temporal graph convolution)
       ->  mean over joints, concatenate 4 x 256, project to 768
       ->  mT5-base decoder
       ->  English sentence
```

The two hands share weights, and body features are added into the hand and face branches.

---

## Repository layout

```
app/          FastAPI backend (main.py, engine.py, run_demo.py) and frontend (static/index.html)
scripts/      fetch_model.py (downloads Uni-Sign code and weights), make_examples.py
evaluation/   evaluation scripts, notebook, results
docs/         Colab notebook for the live demo
```

<details>
<summary><b>Files in <code>evaluation/results/</code></b></summary>

- `main_test_metrics.json`, `main_test_predictions.csv`: the full test run
- `predictions_review.md`: best, random and worst clips, reference next to model output
- `predictions_all.md`: every clip
- `sub_*_predictions.csv`: control runs

</details>

---

## Run it

### Option A: Colab (recommended for a live demo)

1. Open `docs/SignSync_Colab_Demo.ipynb` in [Google Colab](https://colab.research.google.com).
2. Run the cells in order. The first run saves the model to your Google Drive; later runs reuse it.
3. The last cell prints a temporary public link. Open it and upload a video.

The link stops working when the notebook stops.

### Option B: Run locally

Live translation needs the Uni-Sign code and weights (about 3.5 GB), which are not stored in this repo.

```bash
git clone https://github.com/Kashish811/SignSync-asl-translation
cd SignSync-asl-translation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_model.py
uvicorn main:app --app-dir app --port 8000
```

Open <http://127.0.0.1:8000>, upload a short ASL video, and the English translation appears. Pose extraction runs on CPU, so allow time per clip.

### Hosted results page

The [Hugging Face Space](https://huggingface.co/spaces/Kashish811/SignSync-asl-translation) is a static page with saved results and a recorded demo. Live upload does not run there, because the model needs more memory than a free static host provides.

---

## Licence and citation

Uni-Sign code and weights are CC BY-NC 4.0 (non-commercial use only). How2Sign has its own terms; see its website.

```bibtex
@article{li2025unisign,
  title   = {Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author  = {Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal = {arXiv preprint arXiv:2501.15187},
  year    = {2025}
}

@inproceedings{duarte2021how2sign,
  title     = {How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language},
  author    = {Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
  booktitle = {CVPR},
  pages     = {2735--2744},
  year      = {2021}
}
```

<div align="center">

Built by **Kashish Chelwani** · Symbiosis Institute of Technology, Pune

</div>
