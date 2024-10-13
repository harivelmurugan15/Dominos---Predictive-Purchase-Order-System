import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_excel(r"C:\Users\ASUS\Downloads\Pizza_Sale.xlsx")
df.drop(['pizza_id','order_id'],axis=1,inplace=True)

df['total_price'] = df['total_price'].fillna(df['total_price'].mean())

df['pizza_category'] = df['pizza_category'].fillna(df['pizza_category'].mode()[0])
df['pizza_ingredients'] = df['pizza_ingredients'].fillna(df['pizza_ingredients'].mode()[0])
df['pizza_name'] = df['pizza_name'].fillna(df['pizza_name'].mode()[0])
df['pizza_name_id'] = df['pizza_name_id'].fillna(df['pizza_name_id'].mode()[0])

df['order_date'] = pd.to_datetime(df['order_date'])

df['week_number'] = df['order_date'].dt.isocalendar().week

weekly_sales = df.groupby(['week_number','pizza_name_id'])['quantity'].sum().reset_index()

# Get unique pizza_name_id values
unique_pizza_ids = weekly_sales['pizza_name_id'].unique()

# Dictionary to store 91 DataFrames, each with 53 data points
pizza_dfs = {}

# Iterate over each unique pizza_name_id and create separate DataFrames
for pizza_id in unique_pizza_ids:
    # Filter the DataFrame for each pizza_id and ensure 53 data points (weeks)
    pizza_dfs[f'{pizza_id}'] = weekly_sales[weekly_sales['pizza_name_id'] == pizza_id].sort_values(by='week_number')

# Empty dictionary to store predictions
predictions = {}

# Loop over each DataFrame in the dictionary
for pizza_name, df in pizza_dfs.items():
    # Create lag features
    df['lag_1'] = df['quantity'].shift(1)
    df['lag_2'] = df['quantity'].shift(2)
    df['lag_3'] = df['quantity'].shift(3)
    df['lag_4'] = df['quantity'].shift(4)
    df['lag_5'] = df['quantity'].shift(5)

    # Rolling Features
    df['rolling_mean_3'] = df['quantity'].rolling(window=3).mean()
    df['rolling_mean_5'] = df['quantity'].rolling(window=5).mean()

    # Drop NaN values created by shifting and rolling
    df.dropna(inplace=True)

    # Features and target
    X = df[['week_number', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'rolling_mean_3', 'rolling_mean_5']]
    y = df['quantity']

    # Set a random seed for reproducibility
    seed = 42

    # Initialize the XGBRegressor model
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=seed)

    # Cross-validation to stabilize the performance measurement
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(xgb_model, X, y, cv=kfold, scoring='neg_mean_absolute_percentage_error')

    # Output cross-validation results (MAPE is already calculated using sklearn's cross_val_score)
    print(f"Cross-Validation MAPE: {-np.mean(cv_scores) * 100:.2f}%")

    # Train the model on the entire dataset
    xgb_model.fit(X, y)

    # Prepare data for week 54 prediction
    # Create a new DataFrame to hold the data
    week_54_data = {
        'week_number': 54,
        'lag_1': df['quantity'].iloc[-1],
        'lag_2': df['quantity'].iloc[-2],
        'lag_3': df['quantity'].iloc[-3],
        'lag_4': df['quantity'].iloc[-4],
        'lag_5': df['quantity'].iloc[-5],
        'rolling_mean_3': df['rolling_mean_3'].iloc[-1],
        'rolling_mean_5': df['rolling_mean_5'].iloc[-1]
    }

    # Convert to DataFrame for prediction
    week_54_df = pd.DataFrame([week_54_data])

    # Predict the quantity for week 54
    predicted_quantity_week_54 = xgb_model.predict(week_54_df)

    # Output the prediction for week 54
    print(f"Predicted quantity for week 54: {predicted_quantity_week_54[0]}")

    # Store the prediction
    predictions[pizza_name] = round(predicted_quantity_week_54[0])

    # Get predictions on the training set (or use test set if you have one)
    y_pred = xgb_model.predict(X)

    # Calculate MAPE on the entire dataset
    train_mape = mean_absolute_percentage_error(y, y_pred)
    print(f"Training MAPE: {train_mape * 100:.2f}%")

# Convert predictions to a DataFrame for better readability
predictions_df = pd.DataFrame(predictions.items(), columns=['pizza_name_id', 'Predicted Quantity for Week 54'])

predictions_df.to_excel("pred_out.xlsx",index=False)