import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns

# Load data files
@st.cache
def load_data():
    features_path = 'features.csv'
    stores_path = 'stores.csv'
    train_path = 'train.csv'
    test_path = 'test.csv'

    df_features = pd.read_csv(features_path, parse_dates=['Date'])
    df_stores = pd.read_csv(stores_path)
    df_train = pd.read_csv(train_path, parse_dates=['Date'])
    df_test = pd.read_csv(test_path, parse_dates=['Date'])
    
    return df_features, df_stores, df_train, df_test

# Streamlit App Introduction
st.title("Walmart Sales Forecast and Analysis")
st.write("### Introduction")
st.write("This application analyzes and forecasts sales data from Walmart. The dataset, available on Kaggle ([link](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data)), includes weekly sales data, store information, and additional economic indicators.")

# Load Data
df_features, df_stores, df_train, df_test = load_data()

# Display data samples
st.write("### Data Samples")
st.write("#### Train Data")
st.write(df_train.head())
st.write("#### Features Data")
st.write(df_features.head())
st.write("#### Stores Data")
st.write(df_stores.head())

# Merge Data
df_train_full = df_train \
    .merge(df_features.drop(columns=['IsHoliday'], inplace=False), 
           on=['Date', 'Store'], how='inner') \
    .merge(df_stores, on=['Store'], how='inner')

# Add date features
def get_dates(df):
    df1 = df.copy()
    df1['Year'] = df1.Date.dt.year
    df1['Month'] = df1.Date.dt.month
    df1['WeekOfYear'] = np.array(df1.Date.dt.isocalendar().week.astype(int))
    df1['DayOfMonth'] = df1.Date.dt.day
    return df1

df_train_full = get_dates(df_train_full)

# Prepare Data for Forecasting
df_full_c = df_train_full[['Date', 'Weekly_Sales']].groupby(['Date']).mean().copy()
df_full_c.index.freq = 'W-FRI'

df_full_c['lag_1'] = df_full_c['Weekly_Sales'].shift(1)
df_full_c['lag_2'] = df_full_c['Weekly_Sales'].shift(2)
df_full_c['lag_3'] = df_full_c['Weekly_Sales'].shift(3)
df_full_c = df_full_c.dropna()

X = df_full_c[['lag_1', 'lag_2', 'lag_3']]
y = df_full_c['Weekly_Sales']

# Split Data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Forecast
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)

# Streamlit Output
st.write("### Forecast Results")
st.write(f"The forecast model achieved a Mean Squared Error (MSE) of {mse:.2f}. This measures the average squared difference between the actual and forecasted values.")

# Plot Forecast
st.write("#### Forecast vs Actual")
fig, ax = plt.subplots()
ax.plot(y_test.index, y_test.values, label='Actual')
ax.plot(y_test.index, y_pred, label='Forecast', linestyle='--')
ax.legend()
ax.set_title('Forecast vs Actual Weekly Sales')
ax.set_xlabel('Date')
ax.set_ylabel('Weekly Sales')
st.pyplot(fig)

# Additional Data Analysis
st.write("### Data Analysis")

# 1. Weekly Sales Distribution
st.write("#### 1. Weekly Sales Distribution")
st.write("This graph shows the distribution of weekly sales across all stores, highlighting the most common sales values.")
fig, ax = plt.subplots()
df_train_full['Weekly_Sales'].plot(kind='hist', bins=30, ax=ax, color='skyblue', edgecolor='black')
ax.set_title('Distribution of Weekly Sales')
ax.set_xlabel('Weekly Sales')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# 2. Average Weekly Sales by Store
st.write("#### 2. Average Weekly Sales by Store")
st.write("This bar chart displays the average weekly sales for each store, helping to identify the best-performing stores.")
store_sales = df_train_full.groupby('Store')['Weekly_Sales'].mean()
fig, ax = plt.subplots()
store_sales.plot(kind='bar', ax=ax, color='green', edgecolor='black')
ax.set_title('Average Weekly Sales by Store')
ax.set_xlabel('Store')
ax.set_ylabel('Average Weekly Sales')
st.pyplot(fig)

# 3. Weekly Sales Over Time
st.write("#### 3. Weekly Sales Over Time")
st.write("This line chart illustrates the trend of weekly sales over time, providing insights into seasonal patterns.")
fig, ax = plt.subplots()
df_full_c['Weekly_Sales'].plot(ax=ax, color='blue')
ax.set_title('Weekly Sales Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Weekly Sales')
st.pyplot(fig)

# 4. Sales Trend Decomposition
st.write("#### 4. Sales Trend Decomposition")
st.write("Using seasonal decomposition, this plot separates the sales data into trend, seasonal, and residual components.")
results = STL(df_full_c['Weekly_Sales']).fit()
fig = results.plot()
st.pyplot(fig)

# 5. Pair Plot Analysis
st.write("#### 5. Pair Plot Analysis")
st.write("This pair plot visualizes the relationships between key variables, such as sales, temperature, fuel price, CPI, and unemployment.")
pairplot_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
pairplot_data = df_train_full[pairplot_columns].dropna()
fig = sns.pairplot(pairplot_data, diag_kind='kde')
st.pyplot(fig)
