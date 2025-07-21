import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv('data/adult_3.csv')


# Remove rows with '?' in any categorical column
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
for col in cat_cols:
    df = df[df[col] != '?']
# Remove irrelevant workclass
df = df[(df['workclass'] != 'Without-pay') & (df['workclass'] != 'Never-worked')]

# Drop redundant features
df = df.drop(columns=['education'])

# Encode categorical features
cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
encoders = {}
for col in cat_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc
joblib.dump(encoders, 'src/encoder.pkl')

# Features and target
X = df.drop(columns=['income'])
y = df['income']
y = LabelEncoder().fit_transform(y)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'src/scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f'RandomForest Accuracy: {acc:.4f}')
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, 'src/best_model.pkl')
print('✅ Saved best model as src/best_model.pkl')
