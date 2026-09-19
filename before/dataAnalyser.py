from datetime import timedelta
from typing import List

import numpy as np
from before.utility import *
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
        self.tr : List[Dot] = []
        self.adx : List[Dot] = []
        self.dm_minus : List[Dot] = []
        self.dm_plus : List[Dot] = []
        self.tr_sma : List[Dot] = []
        self.tr_ema : List[Dot] = []
        self.dm_plus_sma : List[Dot] = []
        self.dm_plus_ema : List[Dot] = []
        self.dm_minus_sma : List[Dot] = []
        self.dm_minus_ema : List[Dot] = []
        self.dx : List[Dot] = []
        self.dx_sma : List[Dot] = []

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
    
    def getRSILen(self):
        return len(self.rsi)
    
    def getADXLen(self):
        return len(self.adx)
    
    def getQuote(self, k:int):
        if self.getQuoteLen() == 0:
            return None
        return self.quoteBuffer[k]

    def getSMA(self, k:int):
        if self.getSMALen() == 0:
            return None
        return self.movingAverage[k]

    def getEMA(self, k:int):
        if self.getEMALen() == 0:
            return None
        return self.movingAverageExponential[k]
    
    def getRSI(self, k:int):
        if self.getRSILen() == 0:
            return None
        return self.rsi[k]
    
    def getOscillateurStochastique(self, k:int):
        if len(self.oscillateurStochastiqueBuffer) == 0:
            return None
        return self.oscillateurStochastiqueBuffer[k]
    
    def getOscillateurStochastiqueSMA(self, k:int):
        if len(self.oscillateurStochastiqueSMABuffer) == 0:
            return None
        return self.oscillateurStochastiqueSMABuffer[k]
    
    def getOscillateurStochastiqueEMA(self, k:int):
        if len(self.oscillateurStochastiqueEMABuffer) == 0:
            return None
        return self.oscillateurStochastiqueEMABuffer[k]
    
    def getADX(self, k:int):
        if self.getADXLen() == 0:
            return None
        return self.adx[k]
    
    def addQuote(self, quote: Quote):
        self.quoteBuffer.append(quote)

    def getMinMaxPrice(self, period:int):
        try:
            closePrice  = [quote.closePrice for quote in self.quoteBuffer[-period:]]
            return min(closePrice), max(closePrice)
        except(Exception):
            print(closePrice)
            sys.exit(1)
    
    def calculate_Stochastique(self,period:int):   
        if len(self.quoteBuffer) < period:
            return
        current_date = self.quoteBuffer[-1].time
        minPrice, maxPrice = self.getMinMaxPrice(period)
        
        try:
            K_percent = 100* (self.quoteBuffer[-1].closePrice - minPrice)/(maxPrice-minPrice)
            DOT  = Dot(self.quoteBuffer[-1].time,K_percent)
            self.oscillateurStochastiqueBuffer.append(DOT)
        except:
            print("Error in calculate_Stochastique")
            print("minPrice",minPrice,"maxPrice",maxPrice, "date",current_date)

    def calculate_growth_ratio(self,current, previous):
        return abs(current/previous)


    def calculate_SMA(self,data:List[Dot], sma:List[Dot]) -> None:
        AVAILABLE_POINTS = len(data)    
        data_value = [dot.value for dot in data]
        #somme en partant du k eme element
        SOMME = sum(data_value)
        
        LAST_QUOTE_TIME = data[-1].time
        DOT = Dot(LAST_QUOTE_TIME,SOMME/AVAILABLE_POINTS) # dot(time, SMA_k)
        sma.append(DOT)
        
    def calculated_RSI(self, period:int):
        data = np.array([quote.closePrice for quote in self.quoteBuffer[-period-1:-1]])
        data = data[1:] - data[:-1]
        gain = sum(data[data>0])/period
        loss = -1 * sum(data[data<0])/period
        RSI = 100
        if loss != 0:    
            RS = gain/loss
            RSI = 100 - 100/(1+RS)
        DOT = Dot(self.quoteBuffer[-1].time, RSI)
        self.rsi.append(DOT)

    def calculate_TRUE_RANGE(self):
        current_quote = self.quoteBuffer[-1]
        previous_quote = self.quoteBuffer[-2]
        TRUE_RANGE = max(current_quote.highPrice - current_quote.lowPrice, abs(current_quote.highPrice - previous_quote.closePrice), abs(current_quote.lowPrice - previous_quote.closePrice))

        self.tr.append(Dot(current_quote.time, TRUE_RANGE))

    def calculate_DM(self):
        current_quote = self.quoteBuffer[-1]
        previous_quote = self.quoteBuffer[-2]
        UP_MOVE = current_quote.highPrice - previous_quote.highPrice
        DOWN_MOVE = previous_quote.lowPrice - current_quote.lowPrice
        DM_PLUS = UP_MOVE
        DM_MINUS = DOWN_MOVE
        if UP_MOVE <= 0 and UP_MOVE <= abs(DOWN_MOVE):
            DM_PLUS = 0 
        if DOWN_MOVE <= abs(UP_MOVE) and DOWN_MOVE <= 0:
            DM_MINUS = 0
        
        self.dm_minus.append(Dot(current_quote.time, DM_MINUS))
        self.dm_plus.append(Dot(current_quote.time, DM_PLUS))

    def calculate_DX(self, period:int):
        
        smoothed_dm_plus = self.dm_plus_ema[-period:]
        smoothed_dm_minus = self.dm_minus_ema[-period:]
        smoothed_tr = self.tr_ema[-period:]
        smoothed_dm_plus_value = [dot.value for dot in smoothed_dm_plus]
        smoothed_dm_minus_value = [dot.value for dot in smoothed_dm_minus]
        smoothed_tr_value = [dot.value for dot in smoothed_tr]
        
        DI_PLUS =  sum(smoothed_dm_plus_value)/sum(smoothed_tr_value)*100
        DI_MINUS = sum(smoothed_dm_minus_value)/sum(smoothed_tr_value)*100

        DX = abs(DI_PLUS - DI_MINUS)/(DI_PLUS + DI_MINUS)*100
        self.dx.append(Dot(self.quoteBuffer[-1].time, DX))
        
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
        CLOSE_PRICE = data[-1].value
        LAST_QUOTE_TIME = data[-1].time
        EMA = (1-alpha) * PREVIOUS_EMA + alpha * CLOSE_PRICE
        DOT = Dot(LAST_QUOTE_TIME, EMA)
        ema.append(DOT)    

    def calculate_Standard(self, period:int):
        if len(self.quoteBuffer) < period:
            return
        data = self.quoteBuffer[-period:]
        data = [Dot(quote.time, quote.closePrice) for quote in data]
        self.calculate_SMA(data,self.movingAverage)
        self.calculate_EMA(data,self.movingAverage,self.movingAverageExponential)

    def calculate_SMAStochastique(self, d_period:int):
        if self.getOscillateurStochastiqueLen() < d_period:
            return
        data = self.oscillateurStochastiqueBuffer[-d_period:]
        self.calculate_SMA(data,self.oscillateurStochastiqueSMABuffer)
        #self.calculate_EMA(data,self.oscillateurStochastiqueSMABuffer,self.oscillateurStochastiqueEMABuffer)
    
    def calculate_ADX(self,period:int):
        if len(self.quoteBuffer) < 2:
            return
        self.calculate_TRUE_RANGE()
        self.calculate_DM()
        
        alpha = 2/(period+1)

        if len(self.tr) < period:
            return
        

        dm_minus_period = self.dm_minus[-period:]
        self.calculate_SMA(dm_minus_period,self.dm_minus_sma)
        self.calculate_EMA(self.dm_minus,self.dm_minus_sma,self.dm_minus_ema,alpha)
        dm_plus_period = self.dm_plus[-period:]
        self.calculate_SMA(dm_plus_period,self.dm_plus_sma)
        self.calculate_EMA(self.dm_plus,self.dm_plus_sma,self.dm_plus_ema,alpha)
        tr_period = self.tr[-period:]
        self.calculate_SMA(tr_period,self.tr_sma)
        self.calculate_EMA(self.tr,self.tr_sma,self.tr_ema,alpha)

        if len(self.tr_ema) < period:
            return
        self.calculate_DX(period)
        if len(self.dx) < period:
            return
        dx_period = self.dx[-period:]
        self.calculate_SMA(dx_period,self.dx_sma)
        self.calculate_EMA(self.dx,self.dx_sma,self.adx,alpha)            
        
