
# Credit Risk Prediction Project

**Author:** Anish Choudhary

## Project Goals
End-to-end classification system for predicting the likelihood of loan default.

## Structure
- `data/`: Raw and processed datasets.
- `notebooks/`: EDA, Preprocessing, and Modeling workflows.
- `src/`: Reusable python modules for production.
- `app/`: Streamlit dashboard.
- `models/`: Saved `joblib` artifacts.

## Quick Start
1. **Setup**: `pip install -r requirements.txt`
2. **Train**: (Run the notebooks in `notebooks/` folder)
3. **Run App**: `streamlit run app/streamlit_app.py`
4. **Test**: `pytest`

## Docker
```bash
docker build -t credit-risk-app .
docker run -p 8501:8501 credit-risk-app
```
