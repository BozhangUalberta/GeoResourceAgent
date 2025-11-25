import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error as MSE
from langchain_core.tools import tool
import json

# Load Dataset
def load_dataset(choice):
    """
    Load a dataset based on the user's choice.
    """
    if choice == "Injectivity":
        return pd.read_excel("CCUS_Data/1CCUS_injectivity.xlsx")
    elif choice == 'Plume radius':
        return pd.read_excel("CCUS_Data/2CCUS_plume_radius.xlsx")
    elif choice == 'Cost with drill new well':
        return pd.read_excel("CCUS_Data/3CCUS_Cost_drill_new_well.xlsx")
    elif choice == 'Cost with use old well':
        return pd.read_excel("CCUS_Data/4CCUS_Cost_use_old_well.xlsx")
    else:
        raise ValueError("Invalid choice for dataset loading")

# Normalize Data
def normalize_data(df, target_column="Target"):
    """
    Normalize the dataset using min-max normalization.
    """
    df_n = (df - df.min()) / (df.max() - df.min())
    x = df_n.drop(columns=[target_column]).values
    y = df_n[target_column].values
    return x, y, df_n

# Train Model
def train_model(x, y, model_choice, test_size=0.4):
    """
    Train the model based on the user's choice.
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1234)
    
    # Model selection
    models = {
        "MLR": LinearRegression(),
        "Ridge": Ridge(5, max_iter=100000),
        "Lasso": Lasso(0.0001, max_iter=100000),
        "LassoLars": LassoLars(),
        "ElasticNetCV": ElasticNetCV(cv=5, random_state=12),
        "SVR": SVR(kernel="rbf", gamma=0.01),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=1),
        "AdaBoostRegressor": AdaBoostRegressor(n_estimators=800, learning_rate=0.9, random_state=1234),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=1),
        "KNeighbors Regressor": KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=4),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=600, max_features='sqrt', max_depth=15, random_state=1234),
        "Neural network": MLPRegressor(hidden_layer_sizes=(100, 100), activation="tanh", batch_size=128, learning_rate="adaptive", random_state=12, max_iter=2000)
    }
    
    model = models.get(model_choice)
    if model is None:
        raise ValueError("Invalid model choice")
    
    # Train the model
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Predict Values
def predict_values(model, X_test):
    """
    Predict values using the trained model.
    """
    return model.predict(X_test)

# Calculate Metrics
def calculate_metrics(y_test, y_pred):
    """
    Calculate R^2 and Mean Squared Error (MSE) for model evaluation.
    """
    r2 = r2_score(y_test, y_pred)
    mse = MSE(y_test, y_pred)
    return round(r2, 4), round(mse, 4)

@tool("predict_CCUS_Para", return_direct=True)
def predict_CCUS_Para(features: dict, target: str, model_choice: str):
    """
    Predict the target CCUS related value(Injectivity, Plume radius,Cost with drill new well,Cost with use old well) 
    based on user input features and specified model.
    Please match the model with user's input to the model names recognized to the function.
    
    Parameters:
    - features (dict): A dictionary containing feature values (Rate, Depth, Thick, Temp, Perm, Poro, Time, Weight).
    - target (str): The target variable to predict.
    - model_choice (str): The regression model to use.
        You can choose from the following regression models to build the prediction:
        MLR (Multiple Linear Regression)
        Ridge (Ridge Regression)
        Lasso (Lasso Regression)
        LassoLars (Lasso Least Angle Regression)
        ElasticNetCV
        SVR (Support Vector Regressor)
        GradientBoostingRegressor
        AdaBoostRegressor
        DecisionTreeRegressor
        KNeighbors Regressor
        Random Forest Regressor
        Neural network
    
    Returns:
    - float: Predicted target value.
    """
    # Load dataset
    df = load_dataset(target)
    
    # Normalize data
    x, y, df_n = normalize_data(df)
    
    # Train model
    model, _, _ = train_model(x, y, model_choice)
    
    # Create input array from features
    feature_values = np.array([[features['Rate'], features['Depth'], features['Thick'], features['Temp'],
                                features['Perm'], features['Poro'], features['Time'], features['Weight']]])
    
    # Normalize input based on dataset statistics
    feature_values_normalized = (feature_values - df.drop(columns=["Target"]).min().values) / (df.drop(columns=["Target"]).max().values - df.drop(columns=["Target"]).min().values)
    
    # Predict target value
    predicted_value_normalized = model.predict(feature_values_normalized)
    
    # Unnormalize the predicted value
    predicted_value = predicted_value_normalized * (df["Target"].max() - df["Target"].min()) + df["Target"].min()
    
    return json.dumps(predicted_value[0])