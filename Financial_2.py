import streamlit as sl
import yfinance as yf
import openai
from IPython.display import Audio
from pathlib import Path
from credentials import api_key
from gtts import gTTS 
import pandas as pd


client = openai.OpenAI(api_key= api_key)
# define all functions
def get_news_headlines(ticker):
  """
  This function retrieves news headlines for a given stock ticker using yfinance.

  Args:
      ticker (str): The stock ticker symbol (e.g., AAPL, GOOG)

  Returns:
      list: A list of dictionaries containing news headlines and potentially other information.
  """
  try:
    # Get stock info
    stock_info = yf.Ticker(ticker)

    # Get news headlines
    news = stock_info.news
    return news
  except:
    # Handle potential exceptions (e.g., invalid ticker)
    return []

def send_data_and_get_response(prompt):
    response = client.chat.completions.create(
        messages = [
            {"role":"user",
             "content": prompt,
            }],
            model = "gpt-4-1106-preview",
    )
    return response


def get_headlines_for_tickers(tickers):
    headlines = []
    for ticker in tickers:
        news_headlines = get_news_headlines(ticker)
    if news_headlines:
      for headline in news_headlines:
        headlines.append(headline["title"])
        # print(headline["title"])
    else:
      print("No news headlines found for", ticker)
    return "\n".join(headlines)
    

def create_personalized_cast(tickers):
    # Example usage
    prompt = '''Imagine you are a professional financial news reader tasked with recording a podcast segment that provides a concise yet informative overview of the latest developments in the stock market. You have been given the following news headlines related to the stock tickers.

    Your goal is to craft a script for the podcast segment that effectively summarizes the key events, announcements, and implications highlighted in these headlines. The script should be structured as follows:

    Introduction (approx. 15 seconds):
    Provide a brief overview of the sector or industry represented by the given stock tickers.
    Mention any significant market conditions or economic factors that could influence this sector.
    News Summary (approx. 1-2 minutes):
    Concisely highlight the most notable news items and developments mentioned in the headlines.
    Analyze the collective sentiment (positive, negative, or neutral) towards the sector based on these news items.
    Discuss any potential impacts or implications these events could have on the companies or stocks in question.
    Market Context (approx. 30-45 seconds):
    Relate the news summary to broader market trends, economic indicators, or industry-specific factors that could further shape the situation.
    Offer insights or analysis on how these external factors might influence the sector or the stocks mentioned.
    Conclusion (approx. 15-30 seconds):
    Provide a brief closing statement that summarizes the overall situation and its significance for investors or market watchers.
    Throughout the script, maintain a professional and objective tone befitting a financial news reader. 
    Use clear and concise language, avoiding overly technical jargon or unnecessary details. 
    The goal is to deliver a informative yet easily digestible summary suitable for a podcast format. Do not mark out each section explicitly, let the flow take care of breaks between different sections.

    Remember to pace your delivery appropriately, allowing for natural pauses and emphasis where necessary to enhance the listener's comprehension and engagement.
    '''

    headlines_list = get_headlines_for_tickers(tickers)
    user_input = headlines_list + "ticker list: " + ', '.join(tickers)+'.\n'+ prompt
    response = send_data_and_get_response(user_input)
    news_content = response.choices[0].message.content
    # print("ChatGPT:", response)
    myobj = gTTS(text=news_content, lang= 'en', slow=False)
    myobj.save("news.mp3")
    return news_content
    
# Example usage
prompt = '''Imagine you are a professional financial news reader tasked with recording a podcast segment that provides a concise yet informative overview of the latest developments in the stock market. You have been given the following news headlines related to the stock tickers.

Your goal is to craft a script for the podcast segment that effectively summarizes the key events, announcements, and implications highlighted in these headlines. Emphasize the compnies whose tickers have been provided. The script should be structured as follows:

Introduction (approx. 15 seconds):
Provide a brief overview of the sector or industry represented by the given stock tickers.
Mention any significant market conditions or economic factors that could influence this sector.
News Summary (approx. 1-2 minutes):
Concisely highlight the most notable news items and developments mentioned in the headlines.
Analyze the collective sentiment (positive, negative, or neutral) towards the sector based on these news items.
Discuss any potential impacts or implications these events could have on the companies or stocks in question.
Market Context (approx. 30-45 seconds):
Relate the news summary to broader market trends, economic indicators, or industry-specific factors that could further shape the situation.
Offer insights or analysis on how these external factors might influence the sector or the stocks mentioned.
Conclusion (approx. 15-30 seconds):
Provide a brief closing statement that summarizes the overall situation and its significance for investors or market watchers.
Throughout the script, maintain a professional and objective tone befitting a financial news reader. 
Use clear and concise language, avoiding overly technical jargon or unnecessary details. 
The goal is to deliver a informative yet easily digestible summary suitable for a podcast format. Do not mark out each section explicitly, let the flow take care of breaks between different sections.

Remember to pace your delivery appropriately, allowing for natural pauses and emphasis where necessary to enhance the listener's comprehension and engagement.
'''


sl.set_page_config(page_title="Gen-AI Financial News Bulletin", page_icon="📈")

sl.markdown("# Gen-AI News Bulletin Demo")
sl.sidebar.header("Gen-AI News Bulletin Demo")

        
sl.header(
            "Welcome to the AI Based Economic News Snapshot"
        )
#sl.write("#You can chat by Entering your queries ")
ticker_details = pd.read_csv("companylist.csv")
ticker_list = ticker_details['Symbol'].tolist()

#ticker_list = ['IBM', 'MSFT','NVDA','GOOGL', 'JPM']
ticker = sl.multiselect("Please Select Maximum 10 Tickers of your Choice", ticker_details, max_selections = 10)

if sl.button('Run News Podcast'):
   response = create_personalized_cast(ticker)
   sl.write(response)
   
if sl.button('Generate Audio'):
    sl.audio("news.mp3")
