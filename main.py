import streamlit as sp
import yfinance as yf
from datetime import date
import datetime

sp.header(""":green[STOCK PREDICTOR]""", divider='grey')

sp.write(""" ### Our Top Performers""")

def get_ticker(name):
  company = yf.Ticker(name)
  return company

c1 = get_ticker("AAPL")
c2 = get_ticker("MSFT")
c3 = get_ticker("TSLA")


apple = yf.download("AAPL", start="2024-05-01", end="2024-05-09")
microsoft = yf.download("MSFT", start="2024-05-01", end="2024-05-09")
tesla = yf.download("TSla", start="2024-05-01", end="2024-05-09")

data1 = c1.history(period = "3mo")
data2 = c2.history(period = "3mo")
data3 = c3.history(period = "3mo")

sp.write(""" ### Apple """)
sp.write(c1.info['longBusinessSummary'])
sp.write(apple)
sp.line_chart(data1.values)

sp.write(""" ### Microsoft """)
sp.write(c2.info['longBusinessSummary'])
sp.write(microsoft)
sp.line_chart(data2.values)

sp.write(""" ### Tesla """)
sp.write(c3.info['longBusinessSummary'])
sp.write(tesla)
sp.line_chart(data3.values)



import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error



sp.title('Stock Price Predictions')
sp.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
sp.sidebar.info("Created and designed by [Simran Singh](https://github.com/Simran18s)")

def main():
    option = sp.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()



@sp.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



option = sp.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = sp.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = sp.sidebar.date_input('Start Date', value=before)
end_date = sp.sidebar.date_input('End date', today)
if sp.sidebar.button('Send'):
    if start_date < end_date:
        sp.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        sp.sidebar.error('Error: End date must fall after start date')




data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    sp.header('Technical Indicators')
    option = sp.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        sp.write('Close Price')
        sp.line_chart(data.Close)
    elif option == 'BB':
        sp.write('BollingerBands')
        sp.line_chart(bb)
    elif option == 'MACD':
        sp.write('Moving Average Convergence Divergence')
        sp.line_chart(macd)
    elif option == 'RSI':
        sp.write('Relative Strength Indicator')
        sp.line_chart(rsi)
    elif option == 'SMA':
        sp.write('Simple Moving Average')
        sp.line_chart(sma)
    else:
        sp.write('Expoenetial Moving Average')
        sp.line_chart(ema)


def dataframe():
    sp.header('Recent Data')
    sp.dataframe(data.tail(10))



def predict():
    model = sp.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = sp.number_input('How many days forecast?', value=5)
    num = int(num)
    if sp.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    sp.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        sp.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
    main()