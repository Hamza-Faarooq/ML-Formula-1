# ML-To-Predict-Formula-1-Races

# Formula One Race Prediction

This project aims to predict the positions of Formula One drivers in a race based on various factors using machine learning techniques.
 
## Project Overview

The project involves the following steps:
1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training and Evaluation
6. Web Application for Prediction

## Dataset

The dataset includes:
- **20 current F1 drivers**
  - Max Verstappen
  - Sergio Perez
  - Charles Leclerc
  - Carlos Sainz
  - Lewis Hamilton
  - George Russell
  - Lando Norris
  - Daniel Ricciardo
  - Esteban Ocon
  - Fernando Alonso
  - Valtteri Bottas
  - Zhou Guanyu
  - Pierre Gasly
  - Yuki Tsunoda
  - Sebastian Vettel
  - Lance Stroll
  - Niko Hulkenberg 
  - Kevin Magnussen
  - Nicholas Latifi
  - Alexander Albon

- **24 F1 races in a year**
  - Bahrain Grand Prix
  - Saudi Arabian Grand Prix
  - Australian Grand Prix
  - Emilia Romagna Grand Prix
  - Miami Grand Prix
  - Spanish Grand Prix
  - Monaco Grand Prix
  - Azerbaijan Grand Prix
  - Canadian Grand Prix
  - British Grand Prix
  - Austrian Grand Prix
  - French Grand Prix
  - Hungarian Grand Prix
  - Belgian Grand Prix
  - Dutch Grand Prix
  - Italian Grand Prix
  - Singapore Grand Prix
  - Japanese Grand Prix
  - United States Grand Prix
  - Mexico City Grand Prix
  - Brazilian Grand Prix
  - Abu Dhabi Grand Prix
  - Las Vegas Grand Prix
  - Qatar Grand Prix

## Installation

Clone the repository and install the required packages:
```bash
git clone https://github.com/yourusername/f1-ml-project.git
cd f1-ml-project
pip install -r requirements.txt
```

## Data Collection

Ensure your dataset contains the following columns:
- `driver`: Name of the driver
- `race`: Name of the race
- `team`: The team of the driver
- `grid_position`: The starting position of the driver
- `weather`: Weather conditions during the race
- `car_performance`: Performance metrics of the car
- `position`: Final position of the driver

# Example data loading script:
# python
```
import pandas as pd
data = pd.read_csv('f1_race_data.csv')
```

## Data Preprocessing

*Ensure data is clean and well-prepared for analysis:*
# python
```
data.fillna(method='ffill', inplace=True)  # Example of handling missing values
data['weather'] = data['weather'].astype('category').cat.codes  # Encoding categorical data
```

## Exploratory Data Analysis (EDA)

*Analyze the dataset to understand the distributions, correlations, and patterns. Visualization libraries like Matplotlib and Seaborn can be used to create plots:*
# python
```
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(data)
plt.show()
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```

## Feature Engineering

**Create new features or modify existing ones to improve model performance. This can include:**
- **Normalizing or scaling numerical features**
- **Creating interaction terms**
- **Encoding categorical features**

# Example:
# python
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['grid_position', 'car_performance']] = scaler.fit_transform(data[['grid_position', 'car_performance']])
```

## Model Training and Evaluation

# Using a RandomForestRegressor for predicting race positions:
# python
```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

# Split data into features and target variable
```
X = data.drop(columns=['position'])
y = data['position']
```

# Split data into training and testing sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

# Train the model
```
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

# Make predictions
```
y_pred = model.predict(X_test)
```

# Evaluate the model
```
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

