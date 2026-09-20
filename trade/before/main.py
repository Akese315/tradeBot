from before.bot import *
from datetime import datetime, timedelta, date

import matplotlib.pyplot as plt
import numpy as np

import argparse

import torch
from dotenv import load_dotenv
from typing import List
from dotenv import load_dotenv
import before.dataManager as dm
import pandas as pd
import before.AI as ai

load_dotenv()

cursor = None
conn = None
    


ALPHA_VENTAGE_KEY = os.getenv('ALPHA_VENTAGE_KEY_1')
PASSWORD = os.getenv('PSW')
HOST = os.getenv('HOST')
USR = os.getenv('USR')
DATABASE = os.getenv('DATABASE')

#connection to the database
try:
    print("Connecting to the database : ",USR,'@',HOST,"to", DATABASE,"with :", PASSWORD)
    conn = mysql.connector.connect(host=HOST,user=USR,password=PASSWORD,database=DATABASE)
    
except Exception as e:
    print(e)

cursor = conn.cursor()

def startGui(symbol:str):
    bot = Bot(symbol,cursor)

def arrayToQuote(array:List):
    quotes = []
    for i in range(len(array)):
        quote = Quote(array[i][2],array[i][1],array[i][4],array[i][3],array[i][0].strptime("%Y-%m-%d %H:%M:%S"),array[i][5])
        quotes.append(quote)
    return quotes

def createDataset(quoteList : List, symbol):
    dataAnalyser = DataAnalyser(symbol)

    NASDAQ_HOUR_A_DAY = 6
    SMA_EMA_PERIOD = 20*NASDAQ_HOUR_A_DAY #en heures
    K_PERIOD = 14*NASDAQ_HOUR_A_DAY  #en heures
    D_PERIOD = 3*NASDAQ_HOUR_A_DAY #en heures
    data = []
    target = []


    for k in range(len(quoteList)-1):
        quote = quoteList[k]
        nextQuote = quoteList[k+1]
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
        volume = quote.volume

        data.append(torch.tensor([openPrice,highPrice,lowPrice,closePrice,sma,ema,osSMA,os,volume]))
        target.append(torch.tensor([nextQuote.closePrice, nextQuote.highPrice, nextQuote.lowPrice]))
        print(f"\tCreating dataset progress: {int((k+1)/len(quoteList)*100)}%", end="\r")
    return data, target


def trainAI(symbol:str):
    rawData = dm.getLocalTrainingData(symbol)
    data_array = np.array(rawData, dtype=object)    
    beginning_time = datetime.now()
    quotes = arrayToQuote(data_array)
    
    data, target = createDataset(quotes, symbol)
    data = torch.stack(data)
    target = torch.stack(target)
    TrainingDataset = ai.GeneralDataset(data=data, target=target, train_ratio=0.7, test_ratio=0.2)
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