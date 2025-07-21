import pandas as pd
import joblib
import sys

# Usage: python predict.py <age> <workclass> <fnlwgt> <education> <educational-num> <marital-status> <occupation> <relationship> <race> <gender> <capital-gain> <capital-loss> <hours-per-week> <native-country>


def main():
    if len(sys.argv) != 15:
        print('Usage: python predict.py <age> <workclass> <fnlwgt> <education> <educational-num> <marital-status> <occupation> <relationship> <race> <gender> <capital-gain> <capital-loss> <hours-per-week> <native-country>')
        return
    # Parse input
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    values = sys.argv[1:]
    input_dict = {col: [val] for col, val in zip(columns, values)}
    input_df = pd.DataFrame(input_dict)
    input_df['age'] = input_df['age'].astype(int)
    input_df['fnlwgt'] = input_df['fnlwgt'].astype(int)
    input_df['educational-num'] = input_df['educational-num'].astype(int)
    input_df['capital-gain'] = input_df['capital-gain'].astype(int)
    input_df['capital-loss'] = input_df['capital-loss'].astype(int)
    input_df['hours-per-week'] = input_df['hours-per-week'].astype(int)

    # Load encoders and scaler
    encoders = joblib.load('src/encoder.pkl')
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        input_df[col] = encoders[col].transform(input_df[col])
    input_df = input_df.drop(columns=['education'])
    scaler = joblib.load('src/scaler.pkl')
    input_scaled = scaler.transform(input_df)

    # Load model
    model = joblib.load('src/best_model.pkl')
    pred = model.predict(input_scaled)[0]
    print(f'Predicted Income Class: {">50K" if pred == 1 else "<=50K"}')

if __name__ == '__main__':
    main()
