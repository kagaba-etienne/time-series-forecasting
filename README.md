# Time series forecasting

Milan internet-traffic forecasting with ARIMAX, LSTM, and Transformer models.

## Prerequisites

Install [Pipenv](https://pipenv.pypa.io/) on your machine (system package or `pipx`). This project requires **Python 3.12**. Jupyter Lab, `ipykernel`, and the remaining Python dependencies are already listed in the `Pipfile` — do not `pip install` them globally.

## Environment

From the repository root:

```bash
pipenv install
pipenv shell
```

## Jupyter kernel

Register a kernel that points at the Pipenv interpreter so the notebook uses this environment:

```bash
python -m ipykernel install --user --name=time-series-forecasting --display-name "Python (time-series-forecasting)"
```

In Jupyter Lab, select **Python (time-series-forecasting)** as the notebook kernel.

## Jupyter Lab

The notebook lives at [`notebook/timeseries-forecasting.ipynb`](notebook/timeseries-forecasting.ipynb).

After `pipenv shell`:

```bash
jupyter lab
```

Or without entering the shell:

```bash
pipenv run start
```

That script launches Jupyter Lab with `--no-browser --port 8888`.

## Data layout

Raw files are gitignored (`data/*` in `.gitignore`) and must live on disk in this structure. Run [`src/data_prep.py`](src/data_prep.py) to build the processed parquet from `data/raw/cdrs/*.txt`. Training ([`src/train.py`](src/train.py)) reads `data/processed/milan_internet_traffic.parquet`.

```
data/
├── raw/
│   ├── cdrs/                         # daily Telecom Italia CDR dumps
│   │   └── sms-call-internet-mi-YYYY-MM-DD.txt
│   └── grid/
│       └── milano-grid.geojson
└── processed/
    └── milan_internet_traffic.parquet
```

## Training logs

Captured stdout from `python src/train.py` (ARIMAX, LSTM, and Transformer across squares 5161, 5059, and 5259) is in [`logs/train.log`](logs/train.log). Forecast figures are saved under `reports/figures/`.

To re-run training and refresh the log:

```bash
pipenv run python src/train.py 2>&1 | tee logs/train.log
```
