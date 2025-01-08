import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns

# Load data files
@st.cache_data
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

st.write(df_train_full['Weekly_Sales'].describe())

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

st.write(df_train_full.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False))

# 4. Impact of Holiday Weeks on Weekly Sales
st.write("#### 4. Impact of Holiday Weeks on Weekly Sales")
st.write("This analysis aims to understand whether there is a significant increase in sales during holiday weeks.")

# Calculate average sales for holiday and non-holiday weeks
holiday_sales = df_train_full[df_train_full['IsHoliday'] == True]['Weekly_Sales'].mean()
non_holiday_sales = df_train_full[df_train_full['IsHoliday'] == False]['Weekly_Sales'].mean()

# Create bar chart for visualization
fig, ax = plt.subplots()
ax.bar(['Non-Holiday Weeks', 'Holiday Weeks'], [non_holiday_sales, holiday_sales], color=['blue', 'orange'])
ax.set_title('Average Weekly Sales During Holiday vs Non-Holiday Weeks')
ax.set_ylabel('Average Weekly Sales')
st.pyplot(fig)

st.write(f"**Average weekly sales during holiday weeks:** ${holiday_sales:,.2f}")
st.write(f"**Average weekly sales during non-holiday weeks:** ${non_holiday_sales:,.2f}")

# 5. Weekly Sales Patterns by Store Type
st.write("#### 5. Weekly Sales Patterns by Store Type")
st.write("This analysis explores whether different store types (A, B, C) exhibit different weekly sales patterns.")

# Group by store type and date for visualization
store_type_sales = df_train_full.groupby(['Type', 'Date'])['Weekly_Sales'].mean().reset_index()

# Group by store type for average weekly sales
average_sales_by_store_type = df_train_full.groupby('Type')['Weekly_Sales'].mean().reset_index()
average_sales_by_store_type.columns = ['Store Type', 'Average Weekly Sales']

# Create line chart for each store type
fig, ax = plt.subplots()
for store_type in store_type_sales['Type'].unique():
    data = store_type_sales[store_type_sales['Type'] == store_type]
    ax.plot(data['Date'], data['Weekly_Sales'], label=f'Store Type {store_type}')
ax.set_title('Average Weekly Sales by Store Type')
ax.set_xlabel('Date')
ax.set_ylabel('Average Weekly Sales')
ax.legend()
st.pyplot(fig)

# Display the average weekly sales by store type as a table
st.write("#### Average Weekly Sales by Store Type (Table)")
st.write(average_sales_by_store_type)