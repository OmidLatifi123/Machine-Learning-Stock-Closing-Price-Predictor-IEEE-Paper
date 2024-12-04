import pandas as pd

file_path = "prices.csv"  
data = pd.read_csv(file_path)

columns_to_exclude = ['minVolume', 'maxVolume', 'minHigh', 'maxHigh', 'minLow', 'maxLow', 
                      'minClose', 'maxClose', 'minOpen', 'maxOpen']
numeric_columns = [col for col in data.columns if col not in columns_to_exclude and data[col].dtype in ['float64', 'int64']]

correlation_matrix = data[numeric_columns].corr()

close_correlation = correlation_matrix['close']

print("Correlation of 'close' with other features:\n")
print(close_correlation)

top_positive = close_correlation.drop('close').nlargest(3)  
top_negative = close_correlation.drop('close').nsmallest(3)  

print("\nTop positively correlated features:")
print(top_positive)

print("\nTop negatively correlated features:")
print(top_negative)
