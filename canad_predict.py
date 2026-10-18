import pandas as pd
import pandas_datareader.wb as wb
import yfinance as yf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Matplotlib backend:", plt.get_backend())
from dateutil.relativedelta import relativedelta
import support as cntry
import numpy as np
from datetime import datetime
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import torch, sys, time, io, base64
np.random.seed(0)
torch.manual_seed(0)
import requests
from support import plot_2html




def ollama(user_question):
    response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'deepseek-v4.1-flash:cloud',
                'prompt': user_question,
                'stream': False
            }
        )
    variable=(response.json()['response'])
    return variable


def economic_code(country_input):
    dt = datetime.now()
    end = dt.year
    start = end - 25
    country = country_input.strip().title()
    if country not in cntry.country_dict:
        print(f"Country '{country}' not recognized. Please check the name and try again.")
        return(None)
    refined_country = cntry.country_dict[country]
    df_gdp = wb.download(indicator='NY.GDP.MKTP.CD', start=start, end=end, country=refined_country)
    df_gdp_capita = wb.download(indicator='NY.GDP.PCAP.CD', start=start, end=end, country=refined_country)
    df_gdp_growth = wb.download(indicator='NY.GDP.MKTP.KD.ZG', start=start, end=end, country=refined_country)
    df_inflation = wb.download(indicator='FP.CPI.TOTL.ZG', start=start, end=end, country=refined_country)
    df_unemploy = wb.download(indicator='SL.UEM.TOTL.ZS', start=start, end=end, country=refined_country)
    df_population = wb.download(indicator='SP.POP.TOTL', start=start, end=end, country=refined_country)
    df_poverty = wb.download(indicator='SI.POV.DDAY', start=start, end=end, country=refined_country)
    df_life = wb.download(indicator='SP.DYN.LE00.IN', start=start, end=end, country=refined_country)
    df_invest = wb.download(indicator='NE.GDI.TOTL.ZS', start=start, end=end, country=refined_country)
    df_residential = pd.read_csv('WS_SPP_csv_col.csv')

    print(df_residential.head())

    df_gdp_capita = df_gdp_capita.reset_index().drop('country', axis=1)
    df_poverty = df_poverty.reset_index()
    df_poverty = df_poverty.drop(['country'], axis=1)
    df_life = df_life.reset_index()
    df_life = df_life.drop(['country'], axis=1)
    df_gdp_growth = df_gdp_growth.reset_index()
    df_gdp_growth = df_gdp_growth.drop(['country'], axis=1)
    df_inflation = df_inflation.reset_index()
    df_inflation = df_inflation.drop(['country'], axis=1)
    df_population = df_population.reset_index()
    df_population = df_population.drop(['country'], axis=1)
    df_unemploy = df_unemploy.reset_index()
    df_unemploy = df_unemploy.drop(['country'], axis=1)
    df_gdp = df_gdp.reset_index().set_index('year')
    df_invest = df_invest.reset_index()
    df_invest = df_invest.drop(['country'], axis=1)

    df_gdp.rename(columns={'NY.GDP.MKTP.CD': 'GDP'}, inplace=True)
    df_gdp_capita.rename(columns={'NY.GDP.PCAP.CD': 'GDP Per Capita'}, inplace=True)
    df_gdp_growth.rename(columns={'NY.GDP.MKTP.KD.ZG': 'GDP Growth'}, inplace=True)
    df_inflation.rename(columns={'FP.CPI.TOTL.ZG': 'Inflation'}, inplace=True)
    df_unemploy.rename(columns={'SL.UEM.TOTL.ZS': 'Unemployment'}, inplace=True)
    df_population.rename(columns={'SP.POP.TOTL': 'Population'}, inplace=True)
    df_poverty.rename(columns={'SI.POV.DDAY': 'Poverty'}, inplace=True)
    df_invest.rename(columns={'NE.GDI.TOTL.ZS': 'Investment'}, inplace=True)
    df_life.rename(columns={'SP.DYN.LE00.IN': 'Life Expectancy'}, inplace=True)

    df_gdp.index = pd.to_datetime(df_gdp.index, format='%Y', errors='coerce')

    df_gdp_capita.index = pd.to_datetime(df_gdp_capita['year'], format='%Y', errors='coerce')
    df_gdp_capita.drop('year', axis=1, inplace=True)
    df_gdp_growth.index = pd.to_datetime(df_gdp_growth['year'], format='%Y', errors='coerce')
    df_gdp_growth.drop('year', axis=1, inplace=True)
    df_invest.index = pd.to_datetime(df_invest['year'], format='%Y', errors='coerce')
    df_invest.drop('year', axis=1, inplace=True)
    df_inflation.index = pd.to_datetime(df_inflation['year'], format='%Y', errors='coerce')
    df_inflation.drop('year', axis=1, inplace=True)
    df_unemploy.index = pd.to_datetime(df_unemploy['year'], format='%Y', errors='coerce')
    df_unemploy.drop('year', axis=1, inplace=True)
    df_population.index = pd.to_datetime(df_population['year'], format='%Y', errors='coerce')
    df_population.drop('year', axis=1, inplace=True)
    df_poverty.index = pd.to_datetime(df_poverty['year'], format='%Y', errors='coerce')
    df_poverty.drop('year', axis=1, inplace=True)
    df_life.index = pd.to_datetime(df_life['year'], format='%Y', errors='coerce')
    df_life.drop('year', axis=1, inplace=True)

    df_gdp = df_gdp.select_dtypes(exclude=['object', 'string'])
    df_percents = df_gdp.join([df_unemploy, df_invest], how="outer")
    df_economy = df_gdp.join(
        [df_gdp_capita, df_poverty, df_life, df_gdp_growth, df_inflation, df_population, df_unemploy, df_invest],
        how='outer')
    df_economy1 = df_economy[['GDP Per Capita', 'GDP Growth', 'Inflation', 'Unemployment']]
    df_economy2 = df_economy.drop(['GDP Per Capita', 'GDP Growth', 'Inflation', 'Unemployment'], axis=1)
    # df_gdp.plot(xlabel='Time',ylabel='GDP',title='GDP Over Time',grid=True,legend=False,rot=45)
    # df_population.plot(xlabel='Time',ylabel='Population In (tens or hundreds) Millions/Billions',title='Population Over Time',grid=True,legend=False,rot=45)
    # df_economy3 = df_economy.drop(['GDP','Population'], axis=1)
    # df_economy3.plot(title='Economic Data',legend=True,grid=True,rot=45)
    # df_percents.plot(title="Percents",legend=True,grid=True,rot=45)
    economic_chart1 = plot_2html(df_population, "Population", "economic")
    economic_chart2 = plot_2html(df_gdp, "GDP In Trillions/Billions", "economic")
    economic_chart3 = plot_2html(df_life, "Life Expect", "economic")
    economic_chart4 = plot_2html(df_unemploy, "Unemployment", "economic")
    economic_chart5 = plot_2html(df_invest, "Investment", "economic")
    economic_chart6 = plot_2html(df_inflation, "Inflation", "economic")
    return (economic_chart1,economic_chart2,economic_chart3,economic_chart4,economic_chart5,economic_chart6,)
if __name__ == "__main__":
    user_question=input("Welcome to Canad predict. To use the chatbot, simply enter your question. For stock and economic data, type '2'. ")
    if user_question !='2':
        while True:
            try:
                print(GPT_turbo(user_question))
            except:
                print("Switching to Ollama...")
                print(ollama(user_question))
            user_question = input("Do you have any other questions? Enter question. 0 to quit.")
            if user_question == '0':
                break
            elif user_question != '0':
                pass
        sys.exit()
    elif user_question=='2':
        print("Rerouting to commerce...")
        economic_or_fin = input("Global economic data or stock data? 'economic' or 'stock'")
    if economic_or_fin == "stock":
        user_symbol = input("What stock symbol would you like to enter? (NASDAQ/NYSE) For example, type 'AAPL' or 'TSLA'. ")
        stock2_orno = input("Type another stock symbol if you like. If no, type no. If yes, type yes.")
        if stock2_orno == "yes":
            user_symbol2 =input("Type another stock ticker symbol. This will just be graphed alongside the primary stock.")
        else:
            pass
        user_time = input("Do you want a custom graph? If yes, type 'yes'.  "
                          "For the past year, type 'y'. "
                          "For the past month, type 'm'. "
                          "For the most recent quarter, type 'q'. ")
        try:
            if user_time == 'yes':
                user_start = int((input("Please enter the date (YYYY-MM-DD)) the graph to start. ")))
                user_end = int((input("Please enter the date (YYYY-MM-DD) you want the graph to end. ")))
                start = datetime.datetime(user_start)
                end = datetime.datetime(user_end)
                df = yf.download(user_symbol, 'yahoo', start, end)
                try:
                    df2 = yf.download(user_symbol2, 'yahoo', start, end)
                except:
                    pass
                time.sleep(1)
            elif user_time == 'y':
                end = datetime.datetime.today()

                try:
                    start = end.replace(year=end.year - 1)
                except ValueError:
                    start = end.replace(year=end.year - 1, day=28)
                df = yf.download(user_symbol, start, end)
                try:
                    df2 = yf.download(user_symbol2, start, end)
                except:
                    pass
                time.sleep(1)
            elif user_time == 'm':
                end = datetime.datetime.today()
                start = end - relativedelta(months=1)
                df = yf.download(user_symbol, start, end)
                try:
                    df2 = yf.download(user_symbol2, start, end)
                except:
                    pass
                time.sleep(1)
            else:
                end = datetime.datetime.today()
                start = end - relativedelta(months=3)
                df = yf.download(user_symbol,start, end)
                try:
                    df2 = yf.download(user_symbol2,start,end)
                except:
                    pass
                time.sleep(1)
        except Exception as ratelimit:
            print(f"Sorry, the Pandas Datareader is having a slight issue.{ratelimit}")
            user_etf = input(
                "Would you like to see a financial ETF (XLF) or a tech ETF (QQQ)? Please enter the primary ticker symbol.")
            if user_etf == "QQQ":
                df = pd.read_csv('Download Data - FUND_US_XNAS_QQQ.csv', index_col=0, parse_dates=True)
                df2 = pd.read_csv('Download Data - FUND_US_ARCX_XLF.csv', index_col=0, parse_dates=True)
                user_symbol = 'QQQ'
            else:
                df = pd.read_csv('Download Data - FUND_US_ARCX_XLF.csv', index_col=0, parse_dates=True)
                df2 = pd.read_csv('Download Data - FUND_US_XNAS_QQQ.csv', index_col=0, parse_dates=True)
                user_symbol = 'XLF'


        class Stock:
            def __init__(self, symbol, data):
                self.symbol = symbol
                self.data = data

            def percent_change(self):
                return self.data["Close"].pct_change()

            def standard_deviation(self):
                return self.data["Close"].iloc[-10:].std()

        print(df['Close'].head(10))
        stock = Stock(user_symbol, df)
        print(f'Percent change: {Stock.percent_change(stock)}')
        print(f'Standard deviation: {Stock.standard_deviation(stock)}')
        latest_price = float(df['Close'].iloc[-1])

        range_high = float(df['High'].max())
        range_low = float(df['Low'].min())
        average_close = float(np.mean(df['Close']))

        print(f"The average price is ${average_close:.2f}")
        print(f"The low is ${range_low:.2f}")
        print(f"The high is ${range_high:.2f}")
        print(f"The current price of {user_symbol} is ${latest_price:.2f}")
        df.index = pd.to_datetime(df.index)
        print(df.head())
        plt.plot(df['Close'],label=user_symbol,color = 'red')
        try:
            plt.plot(df2['Close'],label = user_symbol2,color = 'blue')
        except:
            pass
        plt.legend()
        plt.grid(True)
        plt.title("Stocks")
        plt.xticks(rotation=45)
        plt.show()
        exit_in=input("Press Enter to exit")

    elif economic_or_fin == "economic":
        economic_code()

def plot_html(df_html,user_symbol):
    fig, ax = plt.subplots()
    ax.plot(df_html['Close'], label=user_symbol, color='red')
    ax.legend()
    ax.grid(True)
    ax.set_title(f"{user_symbol} Stock Price")
    fig.autofmt_xdate()


    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64