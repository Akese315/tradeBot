import requests
import mysql.connector
import time
import os
from datetime import datetime, timedelta, date

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import sys
import torch
import threading
from typing import List
from dotenv import load_dotenv
import before.dataManager as dm
import pandas as pd
import before.AI as ai
from before.utility import *
load_dotenv()


ALPHA_VENTAGE_KEY = os.getenv('ALPHA_VENTAGE_KEY_1')
cursor = None
conn = None
    
#connection to the database
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="myDBAkese315",
        database="TRADEBOT"
    )

except Exception as e:
    print(e)

#global cursor
cursor = conn.cursor()




class DataScrapper:

    BASE_URL = "https://finnhub.io/api/v1"
    URL_QUOTE = "/quote?symbol="
    URL_RECOMMENDATION = "/stock/recommendation?symbol="
    URL_STATUS = "/stock/market-status?exchange=US"
    URL_LOOKUP = "/search?q="

    def __init__(self, symbol : str, max_candle: int ) -> None:
        self.symbol = symbol
        self.MAX_CANDLES =max_candle
        self.apiKey = "cm6m4apr01qg94pu6mpgcm6m4apr01qg94pu6mq0"
        
        self.headers = {
            "X-Finnhub-Token" : self.apiKey
        }

    def getQuote(self):
        url = self.BASE_URL + self.URL_QUOTE + self.symbol
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print(response.status_code)
        else:
            decoded_response = response.json()
            return Quote(decoded_response["o"],decoded_response["pc"],decoded_response["h"],decoded_response["l"],decoded_response["t"])
        
    def isMarketOpen(self):
        url = self.BASE_URL + self.URL_STATUS
        response = requests.get(url, headers=self.headers)
        formatted_response = response.json()
        return formatted_response["isOpen"]

    def exists(self):
        url = self.BASE_URL + self.URL_LOOKUP+self.symbol
        response = requests.get(url,headers=self.headers)
        formatted_response = response.json()
        return formatted_response['count']
    
    def getRecommendation(self):
        url = self.BASE_URL + self.URL_RECOMMENDATION
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print(response.status_code)
        else:
            return json.loads(response.json())


class Bot:
    def __init__(self, symbol : str, cursor) -> None:
        self.dataAnalyser = DataAnalyser()
        self.dataScrapper = DataScrapper(symbol,40000)
        self.symbolExists = False
        if self.dataScrapper.exists() > 0:
            self.symbolExists = True
        self.botSymbol = symbol
        self.running = False
        self.isOpen =  True
        self.cursor = cursor
        self.INSERT_STOCK = "INSERT INTO stocks (symbol, currentPrice, percentChange, changePrice, highPriceDay, lowPriceDay, SMA,EMA,OSC,OSC_EMA) VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s,%s)"


    def update(self):
        if not self.symbolExists:
            return
        isMarketOpen = self.dataScrapper.isMarketOpen()
        if isMarketOpen is False:
            if self.isOpen is True:
                self.isOpen = False
                print(self.botSymbol + " market is closed")
            return
        quote : Quote = self.dataScrapper.getQuote()
        self.dataAnalyser.addQuote(quote)
        self.dataAnalyser.setStochastique()
        self.dataAnalyser.setStandard()
        self.dataAnalyser.setOscillateurStochastique()
    
    def getDatas(self):
        if not self.symbolExists:
            return
        PRICE = [Dot(quote.time, quote.currentPrice) for quote in self.dataAnalyser.quoteBuffer]
        SMA = self.dataAnalyser.movingAverage
        EMA = self.dataAnalyser.movingAverageExponential
        OSC = self.dataAnalyser.oscillateurStochastiqueBuffer
        OSC_EMA = self.dataAnalyser.oscillateurStochastiqueEMABuffer
        return {"price":PRICE, "sma":SMA, "ema":EMA, "osc":OSC, "osc_ema":OSC_EMA}

    def insertInDatabase(self):
        if not self.symbolExists:
            return
        SMA = self.dataAnalyser.getSMA(self.dataAnalyser.getSMALen()-1)
        EMA = self.dataAnalyser.getEMA(self.dataAnalyser.getEMALen()-1)
        OSC = self.dataAnalyser.getOscillateurStochastique(self.dataAnalyser.getOscillateurStochastiqueLen()-1)
        OSC_EMA = self.dataAnalyser.getOscillateurStochastiqueEMA(self.dataAnalyser.getOscillateurStochastiqueEMALen()-1)
        QUOTE = self.dataAnalyser.getQuote(self.dataAnalyser.getQuoteLen()-1)
        self.cursor.execute(self.INSERT_STOCK,(self.botSymbol, QUOTE.currentPrice,QUOTE.percentChange,QUOTE.currentPrice-QUOTE.previousClosePrice,QUOTE.highPriceDay,QUOTE.lowPriceDay,SMA, EMA, OSC, OSC_EMA))


cursor.close()
conn.close()
'''
NVIDIA_BOT = Bot("NVDA",cursor)
BITCOIN_BOT = Bot("BTCUSD",cursor)
ETH_BOT = Bot("ETHUSD",cursor)
plt.title('Graphique en temps réel')
plt.xlabel('Index')
plt.ylabel('Valeur')

while(True):
    
    data_analyzed = NVIDIA_BOT.update()
    conn.commit()
    plt.clf()
    datas = NVIDIA_BOT.getDatas()
    price_time = [dot.time for dot in datas["price"]]
    price_value = [dot.value for dot in datas["price"]]
    SmaArray = [dot.value for dot in datas["sma"]]
    SmaArrayTime = [dot.time for dot in datas["sma"]]
    EmaArray = [dot.value for dot in datas["ema"]]
    EmaArrayTime = [dot.time for dot in datas["ema"]]

    plt.plot(price_time,price_value,'-', color='blue')
    plt.plot(SmaArrayTime,SmaArray,'-', color='red')
    plt.plot(EmaArrayTime,EmaArray,'-', color='green')
    plt.show(block=False)
    plt.pause(10)
'''
