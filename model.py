import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading datasets...")
customer = pd.read_csv('customer.csv')
order = pd.read_csv('order.csv')

# Deduplicate and Merge
customer = customer.drop_duplicates(subset=['customer_id'])
merged = pd.merge(order, customer[['customer_id', 'area']], on='customer_id', how='inner')

# 2. Feature Engineering
print("Engineering features...")
merged['order_datetime'] = pd.to_datetime(merged['promised_delivery_time'])
merged['Hour_of_Day'] = merged['order_datetime'].dt.hour
merged['Day_of_Week'] = merged['order_datetime'].dt.day_name()
merged['area'] = merged['area'].astype(str).str.lower().str.strip()

# 3. Data Transformation
target_encoder = LabelEncoder()
merged['delivery_status_encoded'] = target_encoder.fit_transform(merged['delivery_status'])

features = merged[['Hour_of_Day', 'Day_of_Week', 'area']]
features_encoded = pd.get_dummies(features, columns=['Day_of_Week', 'area'], drop_first=False)

X = features_encoded
y = merged['delivery_status_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Tuning with GridSearchCV
print("Tuning Decision Tree Hyperparameters...")
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# 5. Evaluation
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTuned Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# 6. Feature Importance Visualization
print("Generating Feature Importance Plot...")
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': best_clf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10), palette='viridis')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')
print("Saved feature_importances.png")

# 7. Confusion Matrix Visualization
print("Generating Confusion Matrix Plot...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_encoder.classes_, 
            yticklabels=target_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")

# 8. Sample Prediction
sample_data = {'Hour_of_Day': [18], 'Day_of_Week': ['Friday'], 'area': ['indiranagar']}
sample_df = pd.DataFrame(sample_data)
# Ensure same columns as X
sample_encoded = pd.get_dummies(sample_df, columns=['Day_of_Week', 'area'])
sample_encoded = sample_encoded.reindex(columns=X.columns, fill_value=0)

sample_pred_encoded = best_clf.predict(sample_encoded)
sample_pred = target_encoder.inverse_transform(sample_pred_encoded)
print(f"\nSample Prediction for Friday at 6 PM in Indiranagar: {sample_pred[0]}")

# 9. Save the trained model
import joblib
import os

model = best_clf  # Assigning our best trained model to the generic 'model' variable
filename = 'model.pkl'
joblib.dump(model, filename)

if os.path.exists(filename):
    print(f"\nSuccess: The model has been saved as '{filename}' in the current directory.")
