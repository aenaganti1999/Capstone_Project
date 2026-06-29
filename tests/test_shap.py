import joblib
import shap
import pandas as pd

# Load model
model = joblib.load("artifacts/model/model.joblib")

# Load your X_test
X_test = pd.read_csv("artifacts/data/X_test.csv")
# import or load it exactly how you do during evaluation

explainer = shap.TreeExplainer(model)

sample = X_test.iloc[[0]]

shap_values = explainer.shap_values(sample)

print("Features:")
print(sample)

print("\nSHAP Values:")
feature_names = sample.columns

for feature, value in zip(feature_names, shap_values[0]):
    print(f"{feature}: {value:.4f}")
