# iGaming Player Churn Prediction

A Python project for exploring real bwin Virtual Casino Gambling data and developing player churn prediction models. Work begins with data validation, followed by feature engineering, exploratory analysis, and model evaluation.

## Local data

Keep these source files in `data/raw/`:

- `Codebook_for_Virtual_Casino_Gambling.pdf`
- `RawDataSet1_DemographicsCasinoTXT.zip`
- `RawDataSet2_DailyAggregCasinoTXT.zip`

Raw and processed datasets are excluded from Git. Obtain the source data separately and consult the codebook for variable definitions. Define and document the churn label and observation windows before training models.

## Setup

Using PowerShell from the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `notebooks/01_data_validation.ipynb` in VS Code and select the Python kernel from `venv`.

## Project structure

- `data/raw/` and `data/processed/`: local source and prepared data
- `notebooks/`: validation and exploratory notebooks
- `src/`: reusable Python code
- `sql/`: SQL queries
- `models/`: trained model artifacts
- `reports/figures/`: reports and figures
- `api/` and `dashboard/`: application components
- `tests/`: automated checks

## Project status

Initial setup. Model performance and a production churn definition have not yet been established.
