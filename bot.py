import requests
import mysql.connector
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import json
import threading
from typing import List

url = "https://finnhub.io/api/v1/quote?symbol=NVDA"
apiKey = "cm6m4apr01qg94pu6mpgcm6m4apr01qg94pu6mq0"
headers = {
	"X-Finnhub-Token" : apiKey
}
cursor = None
conn = None
INSERT_STOCK = "INSERT INTO stocks (currentPrice, percentChange, changePrice, highPriceDay, lowPriceDay, MA) VALUES ( %s, %s, %s, %s, %s, %s)"
MOYENNE_N = 100
MAXPOINTSGRAPH = 4000
SmaArray = []
SmaX = []
PriceArray=[]
time_points = []


try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="myDBAkese315",
        database="TRADEBOT"
    )
except Exception as e:
    print(e)

cursor = conn.cursor()

def sumInverse(array,n_end):
    somme =0
    for i in range(len(array)-n_end,len(array)):
        somme += array[i]
    return somme

def addToMoyenneArray(value):
    SmaArray.append(value)

def moyenne_mobile():
    if len(PriceArray)%MOYENNE_N != 0:
        return None
    return sumInverse(PriceArray, MOYENNE_N)/MOYENNE_N
    


def decode(response):
    return {
        "time" : response["t"],
        "currentPrice" : response["c"],
        "percentChange": response["dp"],
        "change" : response["d"],
        "highPriceDay" : response["h"],
        "lowPriceDay" : response["l"]
        }
    
def insertInDatabase(response, MA):
    cursor.execute(INSERT_STOCK,(response["currentPrice"],response["percentChange"],
                                 response["change"],response["highPriceDay"],response["lowPriceDay"], MA))
    conn.commit()

def sendRequest():
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(response.status_code)
    else:
        print(response.json())
        return response.json()

def updateGraphePrice():

    if(len(time_points) > MAXPOINTSGRAPH):
        time_points.pop(0)
    if(len(PriceArray)> MAXPOINTSGRAPH):
        PriceArray.pop(0)    
    plt.plot(time_points,PriceArray,'o-' ,color='green')
   
def updateSMAGraphe():
    plt.plot(SmaX,SmaArray,'o-', color='red')

def initRequests():
    time_started  = datetime.timestamp(datetime.now())
    plt.title('Graphique en temps réel')
    plt.xlabel('Index')
    plt.ylabel('Valeur')
    while(True):
        plt.clf()
        response= sendRequest()
        response_formated = decode(response)
        PriceArray.append(response_formated["currentPrice"])
        time_points.append(response_formated["time"] - time_started)
        MA = moyenne_mobile()
        if MA != None:
            SmaX.append(response_formated["time"] - time_started)
            print(SmaX)
            addToMoyenneArray(MA) 
            print(SmaArray)
        updateSMAGraphe()      
        insertInDatabase(response_formated,MA)
        updateGraphePrice()
        plt.show(block=False)
        plt.pause(1)

class Quote:

    def __init__(self,openPriceDay, previousClosePrice, highPriceDay, lowPriceDay, percentChange,time,currentPrice) -> None:
        self.openPriceDay = openPriceDay
        self.previousClosePrice = previousClosePrice
        self.highPriceDay = highPriceDay
        self.lowPriceDay = lowPriceDay
        self.percentChange = percentChange
        self.currentPrice = currentPrice
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

    def __init__(self, symbol : str, max_candle: int ) -> None:
        self.symbol = symbol
        self.MAX_CANDLES =max_candle
        self.apiKey = "cm6m4apr01qg94pu6mpgcm6m4apr01qg94pu6mq0"
        
        self.headers = {
            "X-Finnhub-Token" : self.apiKey
        }

    def getQuotes(self):
        url = self.BASE_URL + self.URL_QUOTE + self.symbol
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print(response.status_code)
        else:
            return json.loads(response.json())
        
    def isMarketOpen(self):
        url = self.BASE_URL + self.URL_STATUS
        response = requests.get(url, headers=self.headers)
        formatted_response = response.json()
        return formatted_response["isOpen"]
    
    def getRecommendation(self):
        url = self.BASE_URL + self.URL_RECOMMENDATION
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print(response.status_code)
        else:
            return json.loads(response.json())

class DataAnalyser():
    
    def __init__(self) -> None:
        self.quoteBuffer :List[Quote] = []
        self.movingAverage :List[Dot] = []
        self.movingAverageExponential : List[Dot] = []
        self.oscillateurStochastiqueBuffer : List[Dot] = []
        self.highPriceDay
        self.lowPriceDay

    def addQuote(self, quote: Quote):
        self.quoteBuffer.append(quote)
    
    def setOscillateurStochastique(self):   
        dot = Dot(
            self.quoteBuffer[len(self.quoteBuffer)-1].time,(self.quoteBuffer[len(self.quoteBuffer)-1].currentPrice - self.lowPriceDay)/(self.highPriceDay-self.lowPriceDay))
        self.oscillateurStochastiqueBuffer.append(dot)

    def setMovingAverage(self,n) -> None:
        somme = 0
        if len(self.movingAverage) < 40:
            return
        for i in range(len(self.quoteBuffer)-n, len(self.quoteBuffer)):
            somme+=self.quoteBuffer[i].currentPrice
        dot = Dot(self.quoteBuffer[len(self.quoteBuffer)-1].time,somme/n)
        self.movingAverage.append(dot)
        

    def setMovingAverageExponnential(self, n, alpha) -> None:
        if len(self.movingAverageExponential) ==0:
            dot = Dot(self.quoteBuffer[0].time,self.quoteBuffer[0].currentPrice)
            self.movingAverageExponential.append(dot)
        currentEMA = alpha*self.quoteBuffer[len(self.quoteBuffer)-1].currentPrice+(1-alpha)*self.movingAverageExponential[len(self.quoteBuffer)-2].value
        dot = Dot(self.quoteBuffer[len(self.quoteBuffer)-1].time, currentEMA)
        self.movingAverageExponential.append(dot)    





































































































































































    def getMovingAverage():
        return

    def getMovingAverageExponnential():
        return


class BotSymbol:

    symbol : str
    dataScrapper : DataScrapper

    def __init__(self, symbol : str) -> None:
        self.symbol = symbol
        self.dataScrapper = DataScrapper(self.symbol,40000)


    

initRequests()


