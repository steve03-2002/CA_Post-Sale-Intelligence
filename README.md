# Post-Sale Revenue Intelligence Dashboard

A Streamlit dashboard aligned to the uploaded problem analysis, *The Post-Sale Intelligence Gap in CRM*.

## Run on Windows
1. Extract the ZIP.
2. Open the folder that contains `app.py` and `requirements.txt`.
3. Click the File Explorer address bar, type `cmd`, press Enter.
4. Run:
   `python -m pip install -r requirements.txt`
   `python -m streamlit run app.py`
5. Open the local URL printed by Streamlit, typically http://localhost:8501.

## What it models
- Cross-system support, delivery, adoption, renewal and CRM account signals
- Weighted risk index (0-100), explicitly not a calibrated churn probability
- Remaining customer value proxy and estimated value at risk
- Scenario expected retention value, intervention cost, net value and expansion pipeline
- Recommended actions and simulated human-approval queue
- Data quality checks and CSV exports

## Data
Uploads CSV/XLSX. Uses `Combined_Data` from the included synthetic workbook if present, otherwise generates a synthetic 1,000-account sample.

## Limitations
This is a prototype with synthetic data and explicit assumptions. The remaining-value calculation is a proxy, not a validated CLV model. Intervention success and expansion win rates are assumptions. It does not connect to Zoho CRM, create tasks, send outreach, or measure actual renewal outcomes. Those are next integration steps.
