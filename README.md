# Employee Salary Predection

This project predicts whether an employee earns >50K or ≤50K using a machine learning model trained on the Adult dataset (`adult 3.csv`).

🚀 Streamlit App Preview: https://share.streamlit.io/Sagar2006/Employee-Salary-Predection-Model/app.py

## Structure
- `data/adult_3.csv`: Main dataset
- `src/train.py`: Script to train the model
- `src/predict.py`: Script to predict income for a new employee
- `app.py`: Streamlit web app for interactive prediction
- `requirements.txt`: Python dependencies

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python src/train.py
   ```
3. Predict income (CLI):
   ```bash
   python src/predict.py <age> <workclass> <fnlwgt> <education> <educational-num> <marital-status> <occupation> <relationship> <race> <gender> <capital-gain> <capital-loss> <hours-per-week> <native-country>
   # Example:
   python src/predict.py 30 Private 100000 Bachelors 13 Married-civ-spouse Exec-managerial Husband White Male 0 0 40 United-States
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
