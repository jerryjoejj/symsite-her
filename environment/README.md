# Environment setup

| File | Contents |
|---|---|
| [requirements-core.txt](requirements-core.txt) | Feature construction and model use |
| [requirements-analysis.txt](requirements-analysis.txt) | Model comparisons and SHAP attribution |
| [observed_mlher_runtime.json](observed_mlher_runtime.json) | Package versions recorded during the CPU checkpoint-loading check |

Create a Python environment and install the core dependencies:

```bash
python -m pip install -r environment/requirements-core.txt
python -m pip install -e .
python examples/predict.py --check-model
```

The requirement files specify minimum versions. Install `requirements-analysis.txt` for model comparisons and SHAP attribution.
