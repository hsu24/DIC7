# CRISP-DM Linear Regression Streamlit App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crisp-dm-linreg.streamlit.app/)

This repository contains a single-file Streamlit application (`app.py`) that demonstrates the six phases of the Cross-Industry Standard Process for Data Mining (CRISP-DM) using a Linear Regression model.

## Features

- **Business Understanding**: Clearly defined objective and success criteria.
- **Data Understanding**: Displaying data samples, statistical summaries, and scatter plot visualizations of the synthetic data.
- **Data Preparation**: Feature scaling using `StandardScaler` and splitting data into training and test sets.
- **Modeling**: Training an Ordinary Least Squares (OLS) Linear Regression model via `scikit-learn` and displaying true vs. learned parameters.
- **Evaluation**: Calculating model metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score, along with regression line visualizations.
- **Deployment**: Providing an interface for custom predictions on new inputs and allowing users to download the trained model pipeline as a `.joblib` artifact.

## Data Generation
Synthetic data is automatically generated based on user-adjustable parameters from the sidebar:
- `n` samples between 100 and 1000
- Feature `x` drawn from a Uniform(-100, 100) distribution
- Target `y = ax + b + noise` 
- Noise Variance and Random Seed options for reproducibility.

## Installation & Running

Ensure you have Python installed. The required packages are:
- `streamlit`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `joblib`

You can install them via pip:
```bash
pip install streamlit numpy pandas matplotlib scikit-learn joblib
```

To run the app:
```bash
python -m streamlit run app.py
```
