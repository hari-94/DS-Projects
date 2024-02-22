#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[2]:


import yfinance as yf
import pandas as pd

# Define the list of stock symbols
stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "PG"]

# Create an empty DataFrame to store the stock data
stock_data = pd.DataFrame()

# Loop through each stock symbol and fetch historical data
for symbol in stock_symbols:
    # Download historical data
    stock = yf.download(symbol, start="2022-01-01", end="2024-01-01")
    
    # Extract the Adjusted Close prices (you can choose other columns as well)
    stock_data[symbol] = stock["Adj Close"]

# Print the DataFrame
print(stock_data.head())


# In[5]:





# In[33]:


# pip install matplotlib



# In[10]:


pip install plotly


# In[7]:


import matplotlib.pyplot as plt

# Plot the stock prices
stock_data.plot(figsize=(10, 6))
plt.title('Top 5 Grossing Stocks - Historical Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend(loc='upper left')
plt.show()

# Generate a summary of the data
summary_stats = stock_data.describe()
print("\nSummary Statistics:")
print(summary_stats)

# Calculate and print the daily returns
daily_returns = stock_data.pct_change()
print("\nDaily Returns:")
print(daily_returns.head())


# In[13]:


import plotly.graph_objects as go
stock_data = pd.DataFrame()

# Loop through each stock symbol and fetch historical data
for symbol in stock_symbols:
    # Download historical data
    stock = yf.download(symbol, start="2022-01-01", end="2024-01-01")
    
    # Extract the Adjusted Close prices (you can choose other columns as well)
    stock_data[symbol] = stock["Adj Close"]

# Create an interactive plot using Plotly
fig = go.Figure()

for symbol in stock_symbols:
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[symbol], mode='lines', name=symbol))

fig.update_layout(
    title='Top 5 Grossing Stocks - Historical Prices',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Adjusted Close Price'),
    legend=dict(x=0, y=1, traceorder='normal'),
)

# Show the interactive plot
fig.show()

# Generate a summary of the data
summary_stats = stock_data.describe()
print("\nSummary Statistics:")
print(summary_stats)

# Calculate and print the daily returns
daily_returns = stock_data.pct_change()
print("\nDaily Returns:")
print(daily_returns.head())


# In[14]:


stock_data


# In[29]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Acquisition
# Download historical data for SPY from Yahoo Finance
symbol = 'SPY'
data = yf.download(symbol, start='2022-01-01', end='2024-01-01')



# In[25]:


data


# In[28]:


# Create a DataFrame
df = pd.DataFrame(data, index=pd.date_range('2022-01-03', periods=5, freq='B'))

# Define short and long windows for the moving averages
short_window = 2
long_window = 4

# Calculate short-term and long-term SMAs
df['Short_SMA'] = df['Adj Close'].rolling(window=short_window).mean()
df['Long_SMA'] = df['Adj Close'].rolling(window=long_window).mean()

# Generate signals based on the SMA crossover
df['Signal'] = 0
df['Signal'][short_window:] = np.where(df['Short_SMA'][short_window:] > df['Long_SMA'][short_window:], 1, 0)

# Generate buy/sell signals
df['Position'] = df['Signal'].diff()

# Calculate daily returns
df['Daily_Return'] = df['Adj Close'].pct_change()

# Backtesting - Calculate cumulative returns
df['Cumulative_Return'] = (1 + df['Daily_Return'] * df['Position']).cumprod()

# Visualization
plt.figure(figsize=(12, 8))

# Plotting the stock prices
plt.subplot(3, 1, 1)
plt.plot(df['Adj Close'], label='Stock Prices', marker='o')
plt.title('Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Plotting the Moving Averages
plt.subplot(3, 1, 2)
plt.plot(df[['Short_SMA', 'Long_SMA']], label=['Short SMA', 'Long SMA'], marker='o')
plt.title('Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Plotting the Buy/Sell signals
plt.subplot(3, 1, 3)
plt.plot(df.index[df['Signal'] == 1], df['Short_SMA'][df['Signal'] == 1], '^', markersize=10, color='g', label='Buy Signal')
plt.plot(df.index[df['Signal'] == -1], df['Short_SMA'][df['Signal'] == -1], 'v', markersize=10, color='r', label='Sell Signal')
plt.title('Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()

# Display the DataFrame
print(df)


# In[32]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a DataFrame
df = pd.DataFrame(data, index=pd.date_range('2022-01-03', periods=5, freq='B'))

# Define short and long windows for the moving averages
short_window = 2
long_window = 4

# Calculate short-term and long-term SMAs
df['Short_SMA'] = df['Adj Close'].rolling(window=short_window).mean()
df['Long_SMA'] = df['Adj Close'].rolling(window=long_window).mean()

# Generate signals based on the SMA crossover
df['Signal'] = 0
df['Signal'][short_window:] = np.where(df['Short_SMA'][short_window:] > df['Long_SMA'][short_window:], 1, 0)

# Generate buy/sell signals
df['Position'] = df['Signal'].diff()

# Calculate daily returns
df['Daily_Return'] = df['Adj Close'].pct_change()

# Backtesting - Calculate cumulative returns
df['Cumulative_Return'] = (1 + df['Daily_Return'] * df['Position']).cumprod()

# Interactive Visualization with Plotly
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Stock Prices", "Moving Averages", "Buy/Sell Signals"])

# Plotting the stock prices
fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines+markers', name='Stock Prices'), row=1, col=1)

# Plotting the Moving Averages
fig.add_trace(go.Scatter(x=df.index, y=df['Short_SMA'], mode='lines+markers', name='Short SMA'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Long_SMA'], mode='lines+markers', name='Long SMA'), row=2, col=1)

# Plotting the Buy/Sell signals
buy_signals = df[df['Signal'] == 1]
sell_signals = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Short_SMA'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'), row=3, col=1)
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Short_SMA'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'), row=3, col=1)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    xaxis_rangeslider_visible=True,
    title_text="Moving Average Crossover Strategy",
    xaxis_title="Date",
    yaxis_title="Price",
)

# Show the plot
fig.show()

# Display the DataFrame
print(df)

