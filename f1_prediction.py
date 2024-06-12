import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Define drivers, teams, and races
drivers_teams = {
    'Max Verstappen': 'REDBULL',
    'Sergio Perez': 'REDBULL',
    'Charles Leclerc': 'FERRARI',
    'Carlos Sainz': 'FERRARI',
    'Lewis Hamilton': 'MERCEDES',
    'George Russell': 'MERCEDES',
    'Lando Norris': 'McLAREN',
    'Oscar Piastri': 'McLAREN',
    'Esteban Ocon': 'ALPINE',
    'Pierre Gasly': 'ALPINE',
    'Valtteri Bottas': 'KICK SAUBER',
    'Zhou Guanyu': 'KICK SAUBER',
    'Yuki Tsunoda': 'RB',
    'Daniel Ricciardo': 'RB',
    'Lance Stroll': 'ASTON MARTIN',
    'Fernando Alonso': 'ASTON MARTIN',
    'Niko Hulkenberg': 'HAAS',
    'Kevin Magnussen': 'HAAS',
    'Logan Sargeant': 'WILLIAMS',
    'Alexander Albon': 'WILLIAMS'
}

races = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
    "Emilia Romagna Grand Prix", "Miami Grand Prix", "Spanish Grand Prix",
    "Monaco Grand Prix", "Azerbaijan Grand Prix", "Canadian Grand Prix",
    "British Grand Prix", "Austrian Grand Prix", "French Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
    "Italian Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix",
    "United States Grand Prix", "Mexico City Grand Prix", "Brazilian Grand Prix",
    "Abu Dhabi Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix"
]

weather_conditions = ['Sunny', 'Rainy', 'Cloudy']
tire_types = ['Soft', 'Medium', 'Hard']

# Generate synthetic data
np.random.seed(42)
data = {
    'driver': np.random.choice(list(drivers_teams.keys()), 480),
    'race': np.random.choice(races, 480),
    'team': np.random.choice(list(drivers_teams.values()), 480),
    'grid_position': np.random.randint(1, 21, 480),
    'weather': np.random.choice(weather_conditions, 480),
    'car_performance': np.random.uniform(0.5, 1.0, 480),
    'position': np.random.randint(1, 21, 480),
    'pit_stop_time': np.random.uniform(2.5, 5.0, 480),  # Time in seconds
    'tire_choice': np.random.choice(tire_types, 480),
    'driver_skill': np.random.uniform(0.5, 1.0, 480)  # Skill rating
}

df = pd.DataFrame(data)

# Save the synthetic dataset to a CSV file
df.to_csv('f1_race_data_enhanced.csv', index=False)

# Data Preprocessing
# Load the dataset
data = pd.read_csv('f1_race_data_enhanced.csv')

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical features
data['weather'] = data['weather'].astype('category').cat.codes
data['team'] = data['team'].astype('category').cat.codes
data['driver'] = data['driver'].astype('category').cat.codes
data['race'] = data['race'].astype('category').cat.codes
data['tire_choice'] = data['tire_choice'].astype('category').cat.codes

# Split features and target variable
X = data.drop(columns=['position'])
y = data['position']

# Data preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['grid_position', 'car_performance', 'pit_stop_time', 'driver_skill']),
        ('cat', OneHotEncoder(), ['weather', 'team', 'driver', 'race', 'tire_choice'])
    ])

# Model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [3, 4, 5],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model
joblib.dump(best_model, 'f1_race_model_enhanced_gbm.pkl')

# Example prediction with new data (uncomment and modify for real predictions)
# new_data = pd.DataFrame({
#     'driver': [0],  # Replace with actual driver code
#     'race': [0],  # Replace with actual race code
#     'team': [0],  # Replace with actual team code
#     'grid_position': [5],  # Example grid position
#     'weather': [0],  # Example weather condition
#     'car_performance': [0.85],  # Example car performance metric
#     'pit_stop_time': [3.5],  # Example pit stop time
#     'tire_choice': [1],  # Example tire choice code
#     'driver_skill': [0.9]  # Example driver skill rating
# })
# prediction = best_model.predict(new_data)
# print(f"Predicted Position: {prediction[0]}")