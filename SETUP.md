# Setup

Steps to get `read_croissant_data.ipynb` running from scratch.

## 1. Create the virtual environment (uv)

```bash
cd /home/ultra/Documents/download
uv venv .venv
```

## 2. Install dependencies

```bash
uv pip install --python .venv/bin/python pandas pyarrow mlcroissant rdflib numpy ipykernel jupyter
```

## 3. Register the venv as a Jupyter kernel

```bash
.venv/bin/python -m ipykernel install --user --name download-venv --display-name "download (.venv)"
```

## 4. Open the notebook

```bash
.venv/bin/jupyter notebook read_croissant_data.ipynb
```

In Jupyter, select the kernel **"download (.venv)"** (Kernel → Change Kernel) if it isn't picked automatically.

## Optional: run headless (no browser)

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=download-venv \
  --ExecutePreprocessor.timeout=1800 \
  read_croissant_data.ipynb
```

## Notes

- `data/` contains large parquet files (context.parquet ~800MB, train.parquet ~230MB); loading `train`, `validate`, `test`, and `context` fully will use a few GB of RAM.
- Plain `pip install` fails here with `error: externally-managed-environment` (Debian's PEP 668 protection) — always install into `.venv` via `uv pip install --python .venv/bin/python ...`, not system pip.
