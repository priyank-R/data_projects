import joblib
import numpy as np

def load_and_predict(model_file, features):
    # Load the trained model from the file
    rf_reg = joblib.load(model_file)  # Replace "model_file" with your model file path
    print('original array: ')
    print(features)
    print('transformed array')

    # Make predictions
    features_array = np.array(features).reshape(1, -1)
    print(features_array)
    prediction = rf_reg.predict(features_array)
    
    return prediction[0]

# Example usage
model_file = "rf_model.joblib"  # Replace with your model file path
input_features = [52, 4, 19.4]  # Example input features: horsepower, cylinders, acceleration
prediction = load_and_predict(model_file, input_features)
print("Predicted MPG:", prediction)
