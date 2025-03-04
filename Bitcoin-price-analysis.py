import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
x = pd.read_csv('/Users/nupurbachhuka/Downloads/bitcoin2017.csv')
x = x.drop(['unix'], axis=1)

# Convert date column to datetime
x['date'] = pd.to_datetime(x['date'])
x.set_index('date', inplace=True)

# Manual calculation of SMA (Simple Moving Average)
def SMA(data, window):
    return data.rolling(window=window).mean()

x['SMA_14'] = SMA(x['close'], 14)

# Manual calculation of EMA (Exponential Moving Average)
def EMA(data, window):
    return data.ewm(span=window, adjust=False).mean()

x['EMA_14'] = EMA(x['close'], 14)

# Manual calculation of RSI (Relative Strength Index)
def RSI(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

x['RSI'] = RSI(x['close'], 14)

plt.figure(1, figsize=(12, 5))
plt.plot(x.index, x['RSI'], color='purple', label='RSI')
plt.axhline(70, linestyle='dashed', color='red', label='Overbought (70)')
plt.axhline(30, linestyle='dashed', color='green', label='Oversold (30)')
plt.title('RSI Indicator')
plt.xlabel('Date')
plt.ylabel('RSI Value')
plt.legend()
plt.grid(True)
plt.show()

# Manual calculation of Bollinger Bands
def Bollinger_Bands(data, window):
    SMA = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = SMA + (std_dev * 2)
    lower_band = SMA - (std_dev * 2)
    return upper_band, lower_band

x['Bollinger_High'], x['Bollinger_Low'] = Bollinger_Bands(x['close'], 14)

plt.figure(2, figsize=(12, 5))
plt.plot(x.index, x['close'], label='Closing Price', color='blue')
plt.plot(x.index, x['Bollinger_High'], label='Upper Band', linestyle='dashed', color='red')
plt.plot(x.index, x['Bollinger_Low'], label='Lower Band', linestyle='dashed', color='green')
plt.fill_between(x.index, x['Bollinger_Low'], x['Bollinger_High'], color='gray', alpha=0.2)
plt.title('Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Creating meaningful features
x['open-close'] = x['open'] - x['close']
x['low-high'] = x['low'] - x['high']
x['target'] = np.where(x['close'].shift(-1) > x['close'], 1, 0)

# Drop NaN values
x.dropna(inplace=True)

# 2x2 Subplots for Open, High, Low, Close Prices
plt.figure(3)
fig, axes = plt.subplots(2, 2, num=3, figsize=(12, 10))

axes[0, 0].plot(x.index, x['open'], color='orange')
axes[0, 0].set_title('Bitcoin Open Price')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price (USD)')
axes[0, 0].grid(True)

axes[0, 1].plot(x.index, x['high'], color='green')
axes[0, 1].set_title('Bitcoin High Price')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Price (USD)')
axes[0, 1].grid(True)

axes[1, 0].plot(x.index, x['low'], color='red')
axes[1, 0].set_title('Bitcoin Low Price')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Price (USD)')
axes[1, 0].grid(True)

axes[1, 1].plot(x.index, x['close'], color='blue')
axes[1, 1].set_title('Bitcoin Close Price')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Price (USD)')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

plt.figure(4, figsize=(10, 6))
# Select only numeric columns to avoid errors
numeric_x = x.select_dtypes(include=[np.number])

# Generate heatmap with numeric data only
sns.heatmap(numeric_x.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# Feature Selection & Scaling
features = x[['open-close', 'low-high', 'SMA_14', 'EMA_14', 'RSI', 'Bollinger_High', 'Bollinger_Low']]
target = x['target']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-Test Split using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
X_train, X_test, y_train, y_test = None, None, None, None
for train_idx, test_idx in tscv.split(features_scaled):
    X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

# Model Training & Evaluation
models = {'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
          'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'--- {name} Model Performance ---')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('\n')

# Calculate Sharpe Ratio for Trading Performance
def sharpe_ratio(returns, risk_free_rate=0.02):
    return (returns.mean() - risk_free_rate) / returns.std()

x['returns'] = x['close'].pct_change()
print(f'Sharpe Ratio: {sharpe_ratio(x["returns"].dropna()):.2f}')

