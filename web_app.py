
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as gr
from starlette.routing import request_response

from canad_predict import GPT_turbo, ollama, economic_code
from support import plot_2html
import yfinance as yf
import datetime, random, time
from datetime import date
import dateparser
from facts_for_home import surprising_facts
from forex_python.converter import CurrencyRates
import os,requests, json
from sympy import sympify, sin, cos, tan, log, ln, pi, E, sqrt, atan, acos, asin
from dotenv import load_dotenv
from gpt4all import GPT4All
import re, webbrowser
from urllib.parse import quote_plus
import wikipedia
import pandas as pd
import pandas_datareader.wb as wb
import support as cntry
def economic_code(country_input):
    dt = datetime.datetime.now()
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
def search_wikipedia(query):
    try:
        results=wikipedia.search(query)
        page=wikipedia.page(results[0],auto_suggest=False)
        return page.summary
    except Exception as e:
        page=wikipedia.page(e.options[0],auto_suggest=False)
        return page.summary
"""VERY IMPORTANT:
"""
def get_news(topic):
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        print("Guardian API key is missing. Add GUARDIAN_API_KEY to .env.")
        return []

    params = {
        "section": topic,
        "page-size": 10,
        "order-by": "newest",
        "show-fields": "headline,thumbnail",
        "api-key": api_key,
    }

    try:
        res = requests.get(
            "https://content.guardianapis.com/search", params=params, timeout=10
        )
        res.raise_for_status()
        return res.json().get("response", {}).get("results", [])
    except (requests.RequestException, ValueError) as error:
        print(f"Could not load Guardian news: {error}")
        return []

model = GPT4All(
    "mistral-7b-instruct-v0.2.Q4_0.gguf",
    model_path="/Users/sid_asuthker/models_folder/models/mistral",
    device="cpu"
)
#gets flight and weather API keys
load_dotenv()

app = Flask(__name__)

def get_coord(city):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    geo_result = requests.get(geo_url).json()
    if "results" not in geo_result:
        return None, None
    lat = geo_result["results"][0]["latitude"]
    long = geo_result["results"][0]["longitude"]
    return lat, long
def g_weather(city):
    lat,long= get_coord(city)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}&current_weather=true"
    response = requests.get(url)
    wdata = response.json()
    weather = wdata["current_weather"]
    wind = weather['windspeed']
    temp = weather['temperature']
    return wind, temp
def all_var(web_answer=None, graph_img=None, current_close=None):
    data = data_collecter()
    data.update({
        "prediction_text": web_answer,
        "graph_img": graph_img,
        "close": current_close,

    })
    return data


@app.route('/canadomics', methods=['GET', 'POST'])
def economic():
    chart_output1 = None
    chart_output2 = None
    chart_output3 = None
    chart_output4 = None
    chart_output5 = None
    chart_output6 = None
    #set to none for the time being before they are defined
    if request.method == 'POST':
        user_economic_question = request.form['country_question']
        chart_output1,chart_output2,chart_output3,chart_output4,chart_output5,chart_output6=economic_code(user_economic_question)
    return render_template('canadomics.html', chart=chart_output1,chart2=chart_output2,chart3=chart_output3,chart4=chart_output4,chart5=chart_output5,chart6=chart_output6)
def data_collecter():
    global last_day, current_fact, current_fact2, current_fact3, prate, prate2, prate3,prate4,prate5,prate6,prate7, chosen_image, chosen_image2, chosen_image3, chosen_image4

    today_dt = datetime.datetime.today()
    day_ofyear = today_dt.timetuple().tm_yday+2

    # initialize globals if first run
    if 'current_fact' not in globals() or current_fact is None:
        current_fact = current_fact2 = current_fact3 = None
        prate = prate2 = prate3= prate4= prate5= prate6= prate7 = None
        chosen_image = None
        chosen_image2 = None
        chosen_image3 = None
        chosen_image4 = None
        last_day = None

    # update daily data if a new day
    if last_day != day_ofyear:
        num_facts = len(surprising_facts)
        fact_key = random.sample(range(num_facts), 3)
        current_fact = surprising_facts[fact_key[0]]
        current_fact2 = surprising_facts[fact_key[1]]
        current_fact3 = surprising_facts[fact_key[2]]

        # get currency rates
        c = CurrencyRates()
        prate = c.get_rate('USD', 'EUR')
        prate2 = c.get_rate('EUR', 'USD')
        prate3 = c.get_rate('USD', 'CAD')
        prate4 = c.get_rate('USD', 'JPY')
        prate5 = c.get_rate('USD', 'GBP')
        prate6 = c.get_rate('GBP', 'USD')
        prate7 = c.get_rate('USD', 'INR')
        # image of the day
        image_folder = "static/images"
        images = sorted(os.listdir(image_folder))
        li_item = date.today().toordinal() % len(images)
        print(li_item)
        chosen_image, chosen_image2,chosen_image3,chosen_image4 = random.sample(images,4)
        if chosen_image:
            print("Image 1 successfully extracted.")
        else:
            print("Image 1 FAILED.")
        if chosen_image2:
            print("Image 2 successfully extracted.")
        else:
            print("Image 2 FAILED.")
        if chosen_image3:
            print("Image 3 successfully extracted.")
        else:
            print("Image 3 FAILED.")
        if chosen_image4:
            print("Image 4 successfully extracted.")
        else:
            print("Image 4 FAILED.")
        last_day = day_ofyear
    gold = yf.Ticker("GC=F")
    gold_price = gold.info.get('regularMarketPrice')
    oil = yf.Ticker("CL=F")
    oil_price = oil.fast_info.get("last_price")
    print(oil_price)
    # always return a dictionary
    return {
        "selected_fact1": current_fact,
        "selected_fact2": current_fact2,
        "selected_fact3": current_fact3,
        "rate": prate,
        "rate2": prate2,
        "rate3": prate3,
        "rate4": prate4,
        "rate5": prate5,
        "rate6": prate6,
        "rate7": prate7,
        "image": chosen_image,
        "image2": chosen_image2,
        "image3": chosen_image3,
        "image4": chosen_image4,
        "wgold_price": gold_price,
        "woil_price": oil_price
    }
@app.route('/canadculator', methods=['GET', 'POST'])
def calculator():
    adv_oper = {"tan": np.tan,
                "sin": np.sin,
                "cos": np.cos,
                "exp": np.exp,
                "sqrt": np.sqrt,
                "log": np.log
                }
    def logb10(x):
        return log(x,10)
    def refine(equation, i):
        dict = adv_oper.copy()
        dict['x'] = i
        refined = eval(equation, {"__builtins__": None}, dict)
        return refined
    def graph(equation):
        equation=equation.replace("^", "**")
        graph = gr.Figure()
        graph.update_layout(
            xaxis=dict(range=[-100,100]),
            yaxis=dict(range=[-200,200])
        )

        # x**2+8x-16
        x = np.linspace(-100, 100, 700)
        try:
            y = np.array([refine(equation, i) for i in x])
        except:
            return "Invalid"

        graph.add_trace(gr.Scatter(x=x, y=y, mode='lines', name="Graph"))
        graph.update_layout(title="Graph",
                            xaxis_title="x",
                            yaxis_title="y")
        graph_html = graph.to_html(full_html=False)  # just the div, not full HTML page
        return graph_html
    result = ""
    graph_result = ""
    if request.method == "POST":
        expr = request.form.get("math_input")
        graph_expr = request.form.get("user_graph")
        if expr:
            try:
                allowed_operators_dict = {"sin": sin,"cos": cos,"tan": tan,"log": logb10,
                                            "ln": ln,"pi": pi,"π": pi,"E": E,"√": sqrt,
                                            "asin": asin, "acos": acos,"atan": atan,}
                expr = re.sub(r'√(\d+)', r'sqrt(\1)', expr)
                result=sympify(expr,locals=allowed_operators_dict).evalf()
            except:
                result="ERROR"
        elif graph_expr:
            graph_result = graph(graph_expr)

    return render_template("canadculator.html", result=result, graph_result=graph_result)
@app.route('/weather', methods=['GET', 'POST'])
def weather():
    user_city=None
    wind,temp=None,None
    wimage=None
    if request.method == "POST":
        user_city = request.form.get("city")
    wind, temp = g_weather(user_city)
    print("Wind and temp:")
    print(wind, temp)
    print(user_city)
    #g_weather returns the wind and temp for Austin. gets split into wind,temp respectively.
    if temp <=0:
        wimage="cold.png"
    elif temp <=35 and temp>=10:
        wimage="mild.png"
    elif temp>=35:
        wimage="hot.png"
    #determines what image to put based on the temperature

    return render_template('weather.html', wimage=wimage,**all_var(),temp=temp,wind=wind)
@app.route('/', methods=['GET', 'POST'])
def fin_predict():
    web_answer = None
    graph_img = None
    current_close = None
    if request.method == 'POST':
        website_userAI_question = request.form.get('question') #gets the input named "question"
        website_userstock_question = request.form.get('ai_ticker') #gets the input named "ticker"
        if website_userAI_question:
            web_answer = ollama(website_userAI_question)
            print("Ollama is being used...")
        if website_userstock_question:
            end = datetime.datetime.today()
            try:
                start = end.replace(year=end.year - 1) #sets timeframe to 1 year before present
            except ValueError:
                start = end.replace(year=end.year - 1, day=28) #incase its leap year

            print(website_userstock_question)
            stock_df = yf.download(website_userstock_question,start,end) #gets stock data
            if stock_df.empty:
                current_close="[Sorry, it looks like there was an error]"
                graph_img=None
            else:
                current_close = round(stock_df['Close'].iloc[-1].item(),2) #gets most recent stock value
                graph_img = plot_2html(stock_df, website_userstock_question, "stock") #creates a static image of the stock price chart

    articles=get_news("money")
    #
    print(articles)
    return render_template('stocks.html', **all_var(web_answer,graph_img,current_close), articles=articles)
current_fact = None
last_day=None
@app.route('/home', methods=['GET', 'POST'])
def home():
    articles=get_news("world")
    return render_template('home.html',**all_var(),articles=articles)
@app.route('/canad_sengine', methods=['GET', 'POST'])
def canad_sengine():
    out_to_html="Search Google"
    wiki_output="Search Wikipedia"
    news_output="Search News"
    if request.method == 'POST':
        if request.form.get("sengine_question"):
            print("Program started.....")
            sengine_userinput = request.form.get('sengine_question').strip() #gets the input named "sengine_question"
            print(sengine_userinput)
            #prompt is ai_gen cuz ai can talk to ai better than i can
            prompt = f"""
            You are a command parser AI. Your job is to read the user's input and return a JSON object describing the user's intent and the main parameter for that intent. 
    
            Rules:
            1. Only use these intents if applicable: "search", "open_app", "get_weather".
            2. Extract the main parameter for the action:
               - "search": what the user wants to look for
               - "open_app": the application name
               - "get_weather": the city name
            3. Return JSON **only**, nothing else.
            4. Format must always be exactly like:
               {{ "intent": "<intent>", "parameter": "<parameter>" }}
            5. If unsure or if the input is not recognized, set:
               {{ "intent": "", "parameter": "" }}
            6. Do NOT include explanations, extra punctuation, or newlines outside the JSON.
            
            Example 1:
            User input: "Search for cute cats on YouTube"
            Output: {{ "intent": "search", "parameter": "cute cats on YouTube" }}
            
            Example 2:
            User input: "Open Google Chrome"
            Output: {{ "intent": "open_app", "parameter": "Google Chrome" }}
            
            Example 3:
            User input: "What's the weather in New York?"
            Output: {{ "intent": "get_weather", "parameter": "New York" }}
            
            Now parse this input:
            User input: "{sengine_userinput.strip()}"
            """
            ai_dic_output=model.generate(prompt,n_predict=250).strip()
            print("AI raw output:", repr(ai_dic_output))
            if not ai_dic_output:
                out_to_html="unknown"
                print("it was empty")
            if "search" in ai_dic_output:
                param = re.search(r'"parameter"\s*:\s*"([^"]*)"', ai_dic_output)
                print("search")
                param = param.group(1)
                print(param)
                #tool_result=google_search(param)
                #out_to_html=f"Searching for {param},{tool_result}"
                #webbrowser.open(f"https://www.google.com/search?q={param}")
                out_to_html = param
                url=f"https://www.google.com/search?q={out_to_html}"
                webbrowser.open(url)
            elif "open_app" in ai_dic_output:
                print("open_app")
                out_to_html = "search"
                url = f"https://www.google.com/search?q={out_to_html}"
                webbrowser.open(url)
            elif "weather" in ai_dic_output:
                print("get_weather")
                out_to_html = "weather"
                url = f"https://www.google.com/search?q={out_to_html}"
                webbrowser.open(url)
        if request.form.get("wiki_question"):
            query=request.form.get("wiki_question").strip()
            wiki_output = search_wikipedia(query)
        if request.form.get("news_question"):
            news_topic=request.form.get("news_question")
            news_output=get_news(news_topic)

    return render_template("canad_sengine.html", classified=out_to_html, wiki_summary=wiki_output, news_headlines=news_output)

def ai_get_weather(user_text):
    return "test weather"
def agent_classify(user_question):
    print("starting agent_classify......")
    prompt = f"""
    Classify the user's question into one of the following categories: flight, weather, or web_search.
    Return ONLY the category (flight, weather, or web_search) without extra words.

    User question: "{user_question}"
    """
    ai_output=model.generate(prompt,n_predict=256)
    classification=ai_output.strip().lower().replace("\n","").replace("\r","").strip()
    print("finished agent_classify......")
    print(f"classification:{classification}")

    return classification
def ai_search(user_question):
    prompt = f"""
    You are a classifier.
    
    Return ONLY one word.
    No code.
    No explanations.
    No sentences.
    No punctuation.
    If the user asks for books → books
    If the user asks for food → food
    
    User: {user_question}
    Answer:
    """
    ai_output = model.generate(prompt, n_predict=5)
    print("finished ai_search......")
    print(f"extracted word:{ai_output}")
    return ai_output
def google_search(query):
    query=str(query)
    print(query)
    query = query.replace(" ", "+")
    query=quote_plus(query)
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open_new_tab(url)
@app.route("/agent", methods=["GET", "POST"])
def agent_page():
    if request.method == "POST":
        data = request.get_json()
        q = data.get("q", "") #user question
        a=agent_classify(q) #classified by agent. basically flight or weather
        if "flight" in a:
            output="flights"
        elif "weather" in a:
            output="weather today"
        elif "search" in a:
            output=ai_search(q)
        else:
            output="web_search(FAILED)"
        ajson=jsonify({"answer": output}) #jsonified. {"answer": flight} or {"answer": weather}.
        print(ajson)
        google_search(ajson)
        return ajson


    return render_template("agent.html")
    #renders the html VERY IMPORTANT



if __name__ == '__main__':
    app.run(debug=True)
