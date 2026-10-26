# Hugging Face Space (Docker) / any container host. CPU build.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt requirements-model.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-model.txt

COPY --chown=user . .

# Uni-Sign code + checkpoint + mT5-base are downloaded at build time (not stored in this repo)
RUN python scripts/fetch_model.py

ENV UNISIGN_DIR=/home/user/app/Uni-Sign CKPT=/home/user/app/hf_unisign/how2sign_pose_only_slt.pth
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "app", "--host", "0.0.0.0", "--port", "7860"]
