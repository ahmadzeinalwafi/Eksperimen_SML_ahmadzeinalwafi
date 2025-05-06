name: Run Data Preprocessing

on:
  workflow_dispatch:
  push:
    paths:
      - preprocessing/dataset.csv
      - preprocessing/automate_ahmadzeinalwafi.py


jobs:
  preprocess:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pandas numpy scikit-learn mlflow==2.19.0

    - name: Run Preprocessing
      run: python preprocessing/automate_ahmadzeinalwafi.py
