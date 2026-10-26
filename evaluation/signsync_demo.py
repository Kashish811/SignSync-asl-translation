"""Video -> English sentence, using Uni-Sign's own demo pipeline (RTMPose pose extraction + GNN + mT5).
  python signsync_demo.py --repo Uni-Sign --ckpt <ckpt.pth> --video clip.mp4
  python signsync_demo.py --repo Uni-Sign --ckpt <ckpt.pth> --gradio     # needs: pip install gradio (separate env recommended)
Requires: pip install onnxruntime-gpu cuda-toolkit ; cd Uni-Sign/demo/rtmlib-main && pip install -e ."""
import argparse, os, re, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="Uni-Sign"); ap.add_argument("--ckpt", required=True)
ap.add_argument("--video"); ap.add_argument("--gradio", action="store_true"); a = ap.parse_args()
repo, ckpt = os.path.abspath(a.repo), os.path.abspath(a.ckpt)
def translate(video):
    r = subprocess.run([sys.executable, "demo/online_inference.py", "--online_video", os.path.abspath(video), "--finetune", ckpt],
                       cwd=repo, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": repo})
    m = re.search(r"Prediction result is:\s*(.*)", r.stdout)
    if not m: raise RuntimeError("no prediction found.\n" + r.stdout[-1500:] + r.stderr[-1500:])
    return m.group(1).strip()
if a.gradio:
    import gradio as gr
    gr.Interface(lambda v: translate(v), gr.Video(label="ASL clip"), gr.Textbox(label="English translation"), title="SignSync").launch()
else:
    print("Prediction:", translate(a.video))
