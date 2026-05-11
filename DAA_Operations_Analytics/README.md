# DAA Reporting Suite - Quick Start Guide

To run this solution on another laptop, follow these steps:

## 1. Copy Files
Copy the entire `DAA_Stand_Allocation` folder. Ensure you have the following structure:
- `app.py`
- `requirements.txt`
- `api/`
- `engine/`
- `web/` (contains `templates/`)
- `data/` (contains `inputs/` and `outputs/`)
- `.env` (API Keys)

## 2. Setup Library
Open your terminal (Command Prompt or PowerShell) inside the project folder and run:
```bash
pip install -r requirements.txt
```

## 3. Run the Dashboard
Run the following command:
```bash
python app.py
```
Wait for the message: `* Running on http://127.0.0.1:5005`
Open your browser and go to that address.

## 4. Refreshing Data
If you change the raw CSVs in `data/inputs/`, run the analytics engine to update the dashboard:
```bash
python engine/daa_2026_insight_engine_phase1.py
```
Then refresh your browser dashboard.
