# Diamond Price Prediction

An interactive Streamlit app that estimates diamond prices using a
K-Nearest Neighbors regression model and diamond characteristics such as carat,
cut, color, and clarity.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run daimond.py
```

The app reads `diamonds.csv` from the repository root and trains its
preprocessing and regression pipeline on startup.

## Deploy on Streamlit Community Cloud

Create an app from this GitHub repository, select the `main` branch, and set
`daimond.py` as the main file path.
