import pandas as pd

# Load the dataset
df = pd.read_csv('insurance_claims.csv')

# Display the first few rows
print(df.head())

# Get column names and data types
print("Columns and types:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check class balance for the target
print("\nTarget value counts:")
print(df['fraud_reported'].value_counts())

# Drop useless columns
df.drop(['_c39', 'policy_number', 'insured_zip', 'incident_location',
         'policy_bind_date', 'incident_date'], axis=1, inplace=True)

# Encode the target column
df['fraud_reported'] = df['fraud_reported'].map({'Y':1, 'N':0})

# Fix missing values
df['authorities_contacted'] = df['authorities_contacted'].fillna('Unknown')

#Encode categorial columns
cat_cols = df.select_dtypes(include='object').columns
print("Categorical columns:", cat_cols.tolist())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Split the data
from sklearn.model_selection import train_test_split

X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the first model (Logistic Regression)
# from sklearn.linear_model import LogisticRegression
#
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# Try scaling the data to reduce the range
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make prediction on the test set
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# See the actual probabilities
y_proba = model.predict_proba(X_test)
print(y_proba[:5])

# Try Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_pred = rf_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Check feature importance
import pandas as pd
import matplotlib.pyplot as plt

importances = rf_model.feature_importances_
feat_names = X.columns
feat_importances = pd.Series(importances, index=feat_names)

# Top 10 most important features
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Try xtreme Gradient Boosting model
import xgboost as xgb
from sklearn.metrics import accuracy_score
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Plot the ROC (Receiver Operating Characteristic) curve
from sklearn.metrics import roc_curve, auc
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Save model to .pkl
import pickle
with open('xgboost_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Load model from .pkl file
with open('xgboost_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)