FROM python:3.13.9-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY environment/requirements-reproduce.txt /tmp/requirements-reproduce.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.1
RUN python -m pip install --no-cache-dir -r /tmp/requirements-reproduce.txt
WORKDIR /root/capsule/code
CMD ["bash", "run"]
