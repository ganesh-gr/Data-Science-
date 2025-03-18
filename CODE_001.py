import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
# Load weather data from CSV file
df = pd.read_csv('weather.csv')

# Step 2: Data Exploration
# Display the first few rows and dataset summary
print("First 5 Rows of Data:\n", df.head())
print("\nDataset Information:\n")
df.info()  # Displays structure and data types
print("\nStatistical Summary:\n", df.describe())

# Step 3: Data Visualization
# Visualize relationships between MinTemp, MaxTemp, and Rainfall
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.title("Pairplot of Temperature and Rainfall")
plt.show()

# Step 4: Feature Engineering (Ensure 'Date' column exists)
if 'Date' in df.columns:
    # Convert 'Date' to datetime and extract the month
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Handle invalid date formats
    df['Month'] = df['Date'].dt.month
else:
    print("Warning: 'Date' column not found. Skipping month-based analysis.")
    df['Month'] = None  # Assign a placeholder if Date is missing

# Step 5: Data Analysis
# Calculate average maximum temperature by month (only if Month column exists)
if df['Month'].notnull().any():
    monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

    # Step 6: Monthly Average Max Temperature Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o', color='orange')
    plt.xlabel('Month')
    plt.ylabel('Average Max Temperature')
    plt.title('Monthly Average Max Temperature')
    plt.grid(True)
    plt.show()
else:
    print("Month-based analysis skipped due to missing or invalid Date data.")

# Step 7: Rainfall Prediction Using Linear Regression
# Define independent variables (features) and dependent variable (target)
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

# Check for missing values and drop them (if any)
if X.isnull().any().any() or y.isnull().any():
    print("Warning: Missing values detected. Dropping rows with missing data.")
    df = df.dropna(subset=['MinTemp', 'MaxTemp', 'Rainfall'])

# Split the data into training and testing sets (80% train, 20% test)
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Rainfall and calculate Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error for Rainfall Prediction: {mse:.2f}")

# Step 8: Identify Insights
# Example: Identify months with highest and lowest average max temperature
if df['Month'].notnull().any():
    highest_rainfall_month = monthly_avg_max_temp.idxmax()
    lowest_rainfall_month = monthly_avg_max_temp.idxmin()
    print(f"\nHighest Avg Max Temp Month: {highest_rainfall_month}, Lowest Avg Max Temp Month: {lowest_rainfall_month}")
else:
    print("No insights available for month-based analysis due to missing or invalid Date data.")
