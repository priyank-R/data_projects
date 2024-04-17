from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import seaborn as sns
import pandas as pd

def train_and_save_model(model_file):
    # Load data
    data = sns.load_dataset("mpg")
    print('data before cleaning: ', data.shape)
    data = show_and_clean_empty_records(df=data, axis=1)
    print('data after cleaning : ', data.shape)
    
    # Prepare data
    X = data[['horsepower', 'cylinders', 'acceleration']]
    y = data['mpg']

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Remove groups with only one occurrence in the target variable for stratification to not throw the error of single occurance
    grouped = data.groupby('mpg')
    data_filtered = grouped.filter(lambda x: len(x) > 1)
    X = data_filtered[['horsepower', 'cylinders', 'acceleration']]
    y = data_filtered['mpg']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    # Train Random Forest Regression model
    rf_reg = RandomForestRegressor(n_estimators=3, random_state=0)
    rf_reg.fit(X_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(rf_reg, model_file)
    
    # Calculate training accuracy
    train_accuracy = rf_reg.score(X_train, y_train)
    
    # Calculate testing accuracy
    test_accuracy = rf_reg.score(X_test, y_test)
    
    print("Model trained and saved successfully!")
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)

def show_empty_records(df, axis):
    # Check for empty values in the DataFrame
    empty_values = df.isna() | df.isnull() | (df == '')
    # Filter the DataFrame to get records with empty values
    records_with_empty_values = df[empty_values.any(axis=axis)]
    print(records_with_empty_values)

def show_and_clean_empty_records(df, axis):
    # Check for empty values in the DataFrame
    empty_values = df.isna() | df.isnull() | (df == '')
    
    # Filter the DataFrame to get records with empty values
    records_with_empty_values = df[empty_values.any(axis=axis)]
    
    # Print the records with empty values
    print("Records with empty values:")
    print(records_with_empty_values)
    
    # Remove the identified empty records from the DataFrame
    df_cleaned = df[~empty_values.any(axis=axis)]
    
    return df_cleaned

# Example usage
train_and_save_model("rf_model.joblib")
