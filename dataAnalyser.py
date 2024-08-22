from datetime import timedelta
from typing import List

import numpy as np
from utility import *
import sys

class DataAnalyser():
    
    def __init__(self, symbol:str) -> None:
        self.symbol = symbol
        self.quoteBuffer : List[Quote] = []
        self.movingAverage :List[Dot] = []
        self.movingAverageExponential : List[Dot] = []
        self.oscillateurStochastiqueBuffer : List[Dot] = []
        self.oscillateurStochastiqueSMABuffer : List[Dot] = []
        self.oscillateurStochastiqueEMABuffer : List[Dot] = []
        self.rsi : List[Dot] = []

        self.MIN_POINTS_SMA = 20
        self.MIN_POINTS_D_PERCENT_SMA = 20
        self.ALPHA = 2/(self.MIN_POINTS_SMA+1)

        #Example of use
        #data  = [Dot(quote.time, quote.currentPrice) for quote in self.quoteBuffer]
        #self.calculate_SMA(data,self.movingAverage,self.MIN_POINTS_SMA, self.MIN_POINTS_SMA)
        #self.calculate_EMA(data,self.movingAverage,self.movingAverageExponential)

        #data2 = self.oscillateurStochastiqueBuffer
        #self.calculate_SMA(data2,self.oscillateurStochastiqueSMABuffer,self.MIN_POINTS_D_PERCENT_SMA, self.MIN_POINTS_D_PERCENT_SMA)
        #self.calculate_EMA(data2,self.oscillateurStochastiqueSMABuffer,self.oscillateurStochastiqueEMABuffer)

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
    
    def getRSI(self, k:int):
        return self.rsi[k]
    
    def getOscillateurStochastique(self, k:int):
        return self.oscillateurStochastiqueBuffer[k]
    
    def getOscillateurStochastiqueSMA(self, k:int):
        return self.oscillateurStochastiqueSMABuffer[k]
    
    def getOscillateurStochastiqueEMA(self, k:int):
        return self.oscillateurStochastiqueEMABuffer[k]
    
    def addQuote(self, quote: Quote):
        self.quoteBuffer.append(quote)

    def getMinMaxPrice(self, period:int):
        i = -2
        closePrice  = []
        
        try:
            for i in range(-1,-len(self.quoteBuffer),-1):
                closePrice.append(self.quoteBuffer[i].closePrice)
                i-=1
            return min(closePrice), max(closePrice)
        except(Exception):
            print(self.quoteBuffer[-1].time - timedelta(hours=period))
            print(self.quoteBuffer[-1].time)
            print(self.quoteBuffer[i].time)
            print(closePrice)
            print(len(self.quoteBuffer))
            sys.exit(1)
    
    def calculate_Stochastique(self,period:int):   
        current_date = self.quoteBuffer[len(self.quoteBuffer)-1].time
        minPrice, maxPrice = self.getMinMaxPrice(period)
        if len(self.quoteBuffer) > 0:
            try:
                K_percent = 100* (self.quoteBuffer[-1].closePrice - minPrice)/(maxPrice-minPrice)
                DOT  = Dot(self.quoteBuffer[-1].time,K_percent)
                self.oscillateurStochastiqueBuffer.append(DOT)
            except:
                print("Error in calculate_Stochastique")
                print("minPrice",minPrice,"maxPrice",maxPrice, "date",current_date)


    def calculate_SMA(self,data:List[Dot], sma:List[Dot], K) -> None:
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
        
    def calculated_RSI(self, period:int):
        data = np.array([quote.closePrice for quote in self.quoteBuffer[-period-1:-1]])
        data = data[1:] - data[:-1]
        gain = data[data>0].sum()/period
        loss = -data[data<0].sum()/period
        RSI = 100
        if loss != 0:    
            RS = gain/loss
            RSI = 100 - 100/(1+RS)
        DOT = Dot(self.quoteBuffer[-1].time, RSI)
        self.rsi.append(DOT)

    def calculate_EMA(self,data:List[Dot], sma:List[Dot],ema:List[Dot],alpha=None) -> None:
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

    def calculate_Standard(self, period:int):
        data = [Dot(quote.time, quote.closePrice) for quote in self.quoteBuffer]
        self.calculate_SMA(data,self.movingAverage,period)
        self.calculate_EMA(data,self.movingAverage,self.movingAverageExponential)

    def calculate_SMAStochastique(self, d_period:int):
        data = self.oscillateurStochastiqueBuffer
        self.calculate_SMA(data,self.oscillateurStochastiqueSMABuffer,d_period)
        #self.calculate_EMA(data,self.oscillateurStochastiqueSMABuffer,self.oscillateurStochastiqueEMABuffer)
