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
import dataManager as dm
import pandas as pd
import AI as ai
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


class Quote:

    def __init__(self,openPriceDay, closePrice, highPrice, lowPrice,time) -> None:
        self.openPrice = openPriceDay
        self.closePrice = closePrice
        self.highPrice = highPrice
        self.lowPrice = lowPrice
        self.time = time

class Dot:
    def __init__(self,time,value) -> None:
        self.time = time
        self.value = value

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

class DataAnalyser():
    
    def __init__(self, symbol:str) -> None:
        self.symbol = symbol
        self.quoteBuffer : List[Quote] = []
        self.movingAverage :List[Dot] = []
        self.movingAverageExponential : List[Dot] = []
        self.oscillateurStochastiqueBuffer : List[Dot] = []
        self.oscillateurStochastiqueSMABuffer : List[Dot] = []
        self.oscillateurStochastiqueEMABuffer : List[Dot] = []

        self.MIN_POINTS_SMA = 20
        self.MIN_POINTS_D_PERCENT_SMA = 20
        self.ALPHA = 2/(self.MIN_POINTS_SMA+1)

        #Example of use
        #data  = [Dot(quote.time, quote.currentPrice) for quote in self.quoteBuffer]
        #self.setSMA(data,self.movingAverage,self.MIN_POINTS_SMA, self.MIN_POINTS_SMA)
        #self.setEMA(data,self.movingAverage,self.movingAverageExponential)

        #data2 = self.oscillateurStochastiqueBuffer
        #self.setSMA(data2,self.oscillateurStochastiqueSMABuffer,self.MIN_POINTS_D_PERCENT_SMA, self.MIN_POINTS_D_PERCENT_SMA)
        #self.setEMA(data2,self.oscillateurStochastiqueSMABuffer,self.oscillateurStochastiqueEMABuffer)

    def getQuoteLen(self):
        return len(self.quoteBuffer)
    
    def getSMALen(self):
        return len(self.movingAverage)
    
    def getEMALen(self):  
        return len(self.movingAverageExponential)
    
    def getOscillateurStochastiqueLen(self):
        return len(self.oscillateurStochastiqueBuffer)
    
    def getOscillateurStochastiqueSMALen(self):
        return len(self.oscillateurStochastiqueSMABuffer)
    
    def getOscillateurStochastiqueEMALen(self):
        return len(self.oscillateurStochastiqueEMABuffer)
    
    def getQuote(self, k:int):
        return self.quoteBuffer[k]

    def getSMA(self, k:int):
        return self.movingAverage[k]

    def getEMA(self, k:int):
        return self.movingAverageExponential[k]
    
    def getOscillateurStochastique(self, k:int):
        return self.oscillateurStochastiqueBuffer[k]
    
    def getOscillateurStochastiqueSMA(self, k:int):
        return self.oscillateurStochastiqueSMABuffer[k]
    
    def getOscillateurStochastiqueEMA(self, k:int):
        return self.oscillateurStochastiqueEMABuffer[k]
    
    def addQuote(self, quote: Quote):
        self.quoteBuffer.append(quote)
    
    def setStochastique(self, period:int):   
        current_date = self.quoteBuffer[len(self.quoteBuffer)-1].time
        minPrice, maxPrice = dm.getMaxMinData(period=period, symbol=self.symbol, current_date=current_date)
        if len(self.quoteBuffer) > 0:
            K_percent = 100* (self.quoteBuffer[len(self.quoteBuffer)-1].closePrice - minPrice)/(maxPrice-minPrice)
            DOT  = Dot(self.quoteBuffer[len(self.quoteBuffer)-1].time,K_percent)
            self.oscillateurStochastiqueBuffer.append(DOT)


    def setSMA(self,data:List[Dot], sma:List[Dot], K) -> None:
        somme_k = 0
        AVAILABLE_POINTS = len(data)    
        if AVAILABLE_POINTS  < K:
            return
        #somme en partant du k eme element
        for i in range(AVAILABLE_POINTS-K, AVAILABLE_POINTS):
            somme_k+=data[i].value 
        LAST_QUOTE_TIME = data[AVAILABLE_POINTS-1].time
        DOT = Dot(LAST_QUOTE_TIME,somme_k/K) # dot(time, SMA_k)
        sma.append(DOT)
    
        

    def setEMA(self,data:List[Dot], sma:List[Dot],ema:List[Dot],alpha=None) -> None:
        #initialisation de la premiere valeur
        if len(ema) ==0:
            ema.append(sma[0]) #prend pour première valeur la moyenne mobile simple
            return             
        
        if alpha == None:
            alpha = self.ALPHA
            #permet de prendre par defaut self.ALPHA si alpha n'est pas renseigné

        AVAILABLE_POINTS = len(data)
        EMA_LEN = len(ema)
        PREVIOUS_EMA = ema[EMA_LEN-1].value
        CLOSE_PRICE = data[AVAILABLE_POINTS-1].value
        LAST_QUOTE_TIME = data[AVAILABLE_POINTS-1].time
        EMA = (1-alpha) * PREVIOUS_EMA + alpha * CLOSE_PRICE
        DOT = Dot(LAST_QUOTE_TIME, EMA)
        ema.append(DOT)    

    def setStandard(self, period:int):
        data = [Dot(quote.time, quote.closePrice) for quote in self.quoteBuffer]
        self.setSMA(data,self.movingAverage,period)
        self.setEMA(data,self.movingAverage,self.movingAverageExponential)

    def setSMAStochastique(self, d_period:int):
        data = self.oscillateurStochastiqueBuffer
        self.setSMA(data,self.oscillateurStochastiqueSMABuffer,d_period)
        #self.setEMA(data,self.oscillateurStochastiqueSMABuffer,self.oscillateurStochastiqueEMABuffer)

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
def startGui(symbol:str):
    bot = Bot(symbol,cursor)

def arrayToQuote(array:List):
    quotes = []
    for i in range(len(array)):
        quote = Quote(array[i][2],array[i][1],array[i][4],array[i][3],array[i][0].strftime("%Y-%m-%d %H:%M:%S"))
        quotes.append(quote)
    return quotes


def trainAI(symbol:str):
    data = dm.getLocalTrainingData(symbol)
    data_array = np.array(data, dtype=object)    
    beginning_time = datetime.now()
    quotes = arrayToQuote(data_array)
    dataAnalyser = DataAnalyser(symbol)

    NASDAQ_HOUR_A_DAY = 6
    SMA_EMA_PERIOD = 20*NASDAQ_HOUR_A_DAY #en heures
    K_PERIOD = 14*NASDAQ_HOUR_A_DAY  #en heures
    D_PERIOD = 3*NASDAQ_HOUR_A_DAY #en heures
    data = []
    target = []


    for k in range(len(quotes)-1):
        quote = quotes[k]
        nextQuote = quotes[k+1]
        dataAnalyser.addQuote(quote)

        if k+1 >= SMA_EMA_PERIOD: #on attend d'avoir au moins 20 jours de données pour commencer à calculer la moyenne mobile
            dataAnalyser.setStandard(SMA_EMA_PERIOD)
            #print("set standard")
        if k+1 >= K_PERIOD: #on attend d'avoir au moins 14 jours de données pour commencer à calculer l'oscillateur stochastique
            dataAnalyser.setStochastique(K_PERIOD)
            #print("set oscillateur stochastique")
        if dataAnalyser.getOscillateurStochastiqueLen() >= D_PERIOD: #on attend d'avoir au moins 3 jours de données pour commencer à calculer la moyenne mobile de l'oscillateur stochastique
            dataAnalyser.setSMAStochastique(D_PERIOD)
            #print("set SMA oscillateur stochastique")
        if dataAnalyser.getOscillateurStochastiqueSMALen() < D_PERIOD or k+1 < SMA_EMA_PERIOD:
            continue
        
        os = dataAnalyser.getOscillateurStochastique(dataAnalyser.getOscillateurStochastiqueLen()-1).value
        osSMA = dataAnalyser.getOscillateurStochastiqueSMA(dataAnalyser.getOscillateurStochastiqueSMALen()-1).value
        ema = dataAnalyser.getEMA(dataAnalyser.getEMALen()-1).value
        sma = dataAnalyser.getSMA(dataAnalyser.getSMALen()-1).value
        closePrice = quote.closePrice
        openPrice = quote.openPrice
        highPrice = quote.highPrice
        lowPrice = quote.lowPrice
        data.append(torch.tensor([openPrice,highPrice,lowPrice,closePrice,sma,ema,osSMA,os]))
        target.append(torch.tensor([nextQuote.closePrice, nextQuote.highPrice, nextQuote.lowPrice]))
        print(f"\tCreating dataset progress: {int((k+1)/len(quotes)*100)}%", end="\r")

    TrainingDataset = ai.TrainingDataset(data=data, target=target)
    ending_time = datetime.now()
    print("Training dataset length", len(TrainingDataset), "in ", str(ending_time-beginning_time))
    ai.trainModel(TrainingDataset)
    ending_time_training = datetime.now()
    print("Training time in ", str(ending_time_training-ending_time))

def main():
    parser = argparse.ArgumentParser(description="Mon Trading Bot")
    subparsers = parser.add_subparsers(dest='commande', required=True)
    parser_start = subparsers.add_parser('start', help='Exécute la commande start')
    parser_start.add_argument("--nogui", action='store_true', help="Execute la commande sans intefarce graphique")
    parser_start.add_argument("--server",action='store_true', help="Execute la commande en appliquant un server")
    parser_start.add_argument("--symbol",type=str, help="selectionne le symbole", required=True)

    parser_train = subparsers.add_parser('train', help='Exécute la commande train')
    parser_train.add_argument("--symbol",type=str, help="selectionne le symbole", required=True)

    args = parser.parse_args()

    if args.commande == "start":
        if args.nogui:
            print("Exécution sans interface graphique")
            #a faire
        if args.server:
            print(f"Exécution avec le serveur: {args.server}")
            # Ajoute ici le code pour exécuter avec le serveur
        if not args.nogui and not args.server:
            print("Exécution par défaut avec l'interface graphique")
            # Ajoute ici le code pour exécuter avec GUI par défaut
    if args.commande == "train":
        print("Exécution de l'entrainement de l'IA")
        trainAI(args.symbol)

if __name__ == "__main__":
    main()