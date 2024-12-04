import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('prices.csv', low_memory=False, dtype=str)  # Load everything as string
data = data.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, replace non-numeric with NaN

columns_to_exclude = [
    'min Volume', 'maxVolume', 'minHigh', 'maxHigh', 'minLow', 'maxLow',
    'minClose', 'maxClose', 'minOpen', 'maxOpen',
    'normalizedVolume', 'normalizedHigh', 'normalizedLow', 'normalizedClose',
    'normalizedOpen', 'normalizedAvgPrice', 'normalizedRange',
    'volatilityIndex', 'normalizedVolatilityIndex'
]

selected_features = ['avgPrice', 'low', 'high', 'open', 'priceRange']

selected_features = [feat for feat in selected_features if feat in data.columns]

data = data.drop(columns=[col for col in columns_to_exclude if col in data.columns])

X = data[selected_features]
y = data['close']

X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X,  
    y,
      test_size=0.2, 
      random_state=42
)

results = {}

print("Training Linear Regression model...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

results['Linear Regression'] = {'MAE': mae_lr, 'MSE': mse_lr, 'R2': r2_lr}

print("Training Random Forest Regressor model...")
rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

results['Random Forest'] = {'MAE': mae_rf, 'MSE': mse_rf, 'R2': r2_rf}

print("Training K-Nearest Neighbors Regressor model...")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

results['KNN Regressor'] = {'MAE': mae_knn, 'MSE': mse_knn, 'R2': r2_knn}

print("Training Decision Tree Regressor model...")
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

results['Decision Tree'] = {'MAE': mae_dt, 'MSE': mse_dt, 'R2': r2_dt}

print("\nModel Performance:")
for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
    print(f"R-squared (R2): {metrics['R2']:.4f}")
