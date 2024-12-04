Abstract

This paper investigates the use of machine learning regression models to predict stock market closing (close) prices using large amounts of stock market data from S&P 500 companies. The prices.csv dataset contains over 850,000 rows of stock price data spanning from 2010 to 2016, with features such as opening price, daily highs and lows, trading volume, and closing price. Through feature engineering new metrics were introduced such as average trading price (avgPrice), price range (priceRange), and volatility index (volatilityIndex) to enhance predictive accuracy, with both normalized and non-normalized versions of all numeric columns/features.
	Four regression models were used to train and evaluate the model in terms of its successful prediction of the closing prices of stocks. These 4 regression models are: Linear Regression, Random Forest Regressor, K-Nearest Neighbors (KNN), and Decision Tree Regressor. The data split consisted of 80% of the data being used to train the model whilst 20% remained for testing. Performance was assessed using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) metrics. Models trained on normalized features consistently outperformed those trained on non-normalized data, achieving lower MAE and MSE values. This is due to non-normalized data being prone to being skewed by larger scale features, such as trading volume (volume). Linear Regression emerged as the most accurate model, followed by Random Forest and KNN, while Decision Trees offered interpretability but slightly higher errors.
	Results highlight the critical role of feature preprocessing, particularly normalization, in improving model performance. Near-perfect R² values (0.9999) across all models demonstrate the strength of the selected features in capturing variability in the target variable. This study shows the potential of machine learning for financial forecasting and highlights areas for improvement, such as incorporating external factors, time-series features, and expanding dataset scope. These findings pave the way for accessible, AI-driven tools for everyday traders and analysts, enhancing decision-making in dynamic financial markets and leveling the playing field for all.
