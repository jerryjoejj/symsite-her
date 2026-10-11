# Runtime environment

Use Python 3.13.9, CPU PyTorch 2.12.1 and the pins in `requirements-reproduce.txt`:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1
python -m pip install -r environment/requirements-reproduce.txt
```

`tested_environment.json` records the isolated validation environment. `requirements-lock.txt` contains its full pip package set. Default analysis uses two CPU threads; neural training uses four.

`postInstall` provides capsule setup commands. The Dockerfile is a local recipe: from the exported capsule root, run `docker build -f environment/Dockerfile -t symsite-her .`. Mount capsule `code/`, `data/` and writable results at `/root/capsule/code`, `/root/capsule/data` and `/root/capsule/results`.
