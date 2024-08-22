import argparse
import sys
from typing import List
from dataAnalyser import DataAnalyser
import requests
import os
from time import sleep
import pandas as pd
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
import requests
import mysql.connector

from utility import Quote

load_dotenv()
ALPHA_VENTAGE_KEY = os.getenv('ALPHA_VENTAGE_KEY_2')
PASSWORD = os.getenv('PSW')
HOST = os.getenv('HOST')
USR = os.getenv('USR')
DATABASE = os.getenv('DATABASE')

#connection to the database
conn = None
cursor = None
try:
    print("Connecting to the database : ",USR,'@',HOST,"to", DATABASE,"with :", PASSWORD)
    conn = mysql.connector.connect(host=HOST,user=USR,password=PASSWORD,database=DATABASE)
    
except Exception as e:
    print(e)

cursor = conn.cursor()




def getMaxMinData(period:int, symbol:str, current_date:str, cursor) -> tuple[float,float]:
    #print(current_date, type(current_date))
    date_period = datetime.strptime(current_date,'%Y-%m-%d %H:%M:%S') - timedelta(days=period)
    SQL_SELECT = "SELECT closePrice FROM training_data_hour WHERE time > '"+date_period.strftime("%Y-%m-%d")+"'AND time <='"+current_date+"' AND symbol='"+symbol+"';"
    #print(SQL_SELECT)
    cursor.execute(SQL_SELECT)
    rows = cursor.fetchall()
    minPrice = float(min(rows)[0])
    maxPrice = float(max(rows)[0])
    
    #print("Lowest and highest price in a period of",period,"days are :",minPrice,"and",maxPrice)

    return (minPrice, maxPrice)

def getLocalTrainingData(symbol:str, cursor) -> List:
    SQL_SELECT = "SELECT time, closePrice, openPrice, lowPrice, highPrice, volume FROM training_data_hour WHERE symbol='"+symbol+"' ORDER BY time ASC;"
    cursor.execute(SQL_SELECT)
    rows = cursor.fetchall()
    return rows

def cleanTrainingData(symbol:str, year:str, cursor, conn):
    SQL_DELETE = "DELETE FROM training_data_hour WHERE symbol='"+symbol+"';"
    if year is not None:
        SQL_DELETE = "DELETE FROM training_data_hour WHERE symbol='"+symbol+"'AND YEAR(time)="+year+";"
    print(SQL_DELETE)
    try:
        cursor.execute(SQL_DELETE)
        conn.commit()
        print("successfully delete",symbol)
    except Exception as e:
            print(e)

def training_data_from_csv(symbol:str, start:str, end:str):
    files_name = []
    for i in range(int(start),int(end)+1):
        year = str(i)
        if i < 10:
            year = "0"+str(i)   
        for k in range(1,13):
            month = str(k)
            if k < 10:
                month = "0"+str(k)   
            files_name.append(symbol+"_"+year+"-"+month+".csv") 
    dataframes = []
    for i in range(len(files_name)):
        file = open("./dataframe/"+files_name[i],mode="r")
        dataframes.append(pd.read_csv(file))
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.sort_values(by="time")
    return combined_df

def getTrainingData(month:str,symbol:str,interval:str, csv=False):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY'
    url+='&symbol='+symbol
    url+='&interval='+interval
    url+='&month='+month
    url+='&apikey='+ALPHA_VENTAGE_KEY
    url+='&outputsize=full'
    dataframe = None
    response = requests.get(url)
    json_parsed_response = response.json()
    metadata = json_parsed_response["Meta Data"]
    print(metadata)
    data = json_parsed_response["Time Series (60min)"]
    items = list(data.items())
    reversed_items = items[::-1]
    reversed_data = dict(reversed_items)

    if csv:
        dataframe = pd.DataFrame(columns=['time', 'symbol', 'highPrice',"lowPrice","closePrice","openPrice","volume"])
        

    for timestamp, values in reversed_data.items():
        open_price = values["1. open"]
        high_price = values["2. high"]
        low_price = values["3. low"]
        close_price = values["4. close"]
        volume = values["5. volume"]
        if csv:
            row = {
                'time': timestamp,
                'symbol': symbol,
                'highPrice': high_price,
                'lowPrice': low_price,
                'closePrice': close_price,
                'openPrice': open_price,
                'volume': volume
            }
            row_dataframe = pd.DataFrame([row])
            dataframe = pd.concat([dataframe, row_dataframe], ignore_index=True)
        try:
            SQL_INSERT = "INSERT INTO training_data_hour (time, symbol, highPrice,lowPrice,closePrice,openPrice,volume) VALUES (%s, %s, %s, %s, %s, %s, %s);"
            cursor.execute(SQL_INSERT,(timestamp,symbol,high_price,low_price,close_price,open_price,volume))
            conn.commit()
        except Exception as e:
            print(e)
    if csv:
        file = open("./dataframe/"+symbol+"_"+month+".csv",mode="w", newline='')
        dataframe.to_csv(file, index=False)
        file.close()

def harvestYear(year:str,symbol:str,interval="60min", csv=False):
    if interval is None:
        interval = "60min"

    now = datetime.now()
    max_month = 12
    if year == str(now.year):
        max_month = now.month
    for i in range(1,max_month+1):   
        date = year
        if i < 10:
            date += "-0"+str(i)
        else:
            date += "-"+str(i)
        getTrainingData(month=date,symbol=symbol, interval=interval,csv=csv)
        sleep(15)

def analyse(symbol:str,start:str,end:str, sma_period):
    dataAnalyser = DataAnalyser(symbol=symbol)
    combined_df = training_data_from_csv(symbol,start,end)
    dataframe = pd.DataFrame(columns=['time', 'symbol', 'highPrice',"lowPrice","closePrice","openPrice","volume"])
    D_PERIOD = 18
    for index, row in combined_df.iterrows():
        
        quote = Quote(row["openPrice"],row["closePrice"],row["highPrice"],row["lowPrice"],datetime.strptime(row["time"],"%Y-%m-%d %H:%M:%S"),row["volume"])
        dataAnalyser.addQuote(quote)
        if index+1 < sma_period:
            continue
        dataAnalyser.calculate_Standard(sma_period)
        dataAnalyser.calculate_Stochastique(sma_period)
        dataAnalyser.calculated_RSI(14)
        sma = dataAnalyser.getSMA(-1)
        ema = dataAnalyser.getEMA(-1)
        rsi = dataAnalyser.getRSI(-1)
        os = dataAnalyser.getOscillateurStochastique(-1)
        if dataAnalyser.getOscillateurStochastiqueLen() < D_PERIOD: #on attend d'avoir au moins 3 jours de données pour commencer à calculer la moyenne mobile de l'oscillateur stochastique
           continue
        dataAnalyser.calculate_SMAStochastique(D_PERIOD)
        sma_os = dataAnalyser.getOscillateurStochastiqueSMA(-1)
        
        row = {
                'time': quote.time,
                'symbol': symbol,
                'highPrice': quote.highPrice,
                'lowPrice': quote.lowPrice,
                'closePrice': quote.closePrice,
                'openPrice': quote.openPrice,
                'volume': quote.volume,
                'sma': sma.value,
                'ema': ema.value,
                'rsi':rsi.value,
                'oscillateurStochastique': os.value,
                'oscillateurStochastiqueSMA': sma_os.value
            }  
        row_dataframe = pd.DataFrame([row])
        dataframe = pd.concat([dataframe, row_dataframe], ignore_index=True)
        print(f"\tCreating dataset progress: {int((index+1)/combined_df.size*100)}%", end="\r")
    file = open("./analysed_dataframe/"+symbol+"_"+start+"-"+end+".csv",mode="w", newline='')
    dataframe.to_csv(file, index=False)
    file.close()
def main():
    parser = argparse.ArgumentParser(description="Mon Data Manager")
    subparsers = parser.add_subparsers(dest='commande', required=True)

    parser_delete = subparsers.add_parser('delete', help='Exécute la commande delete')
    parser_delete.add_argument("--symbol",type=str,required=True, help="Indiquer le symbole à supprimer")
    parser_delete.add_argument("--year",type=str,required=False,help="Indiquer l'annee à ajouter")

    parser_harvest = subparsers.add_parser('harvestYear', help='Exécute la commande harvestYear')
    parser_harvest.add_argument("--symbol",type=str,required=True,help="Indiquer le symbole à ajouter")
    parser_harvest.add_argument("--year",type=str,required=True,help="Indiquer l'annee à ajouter")
    parser_harvest.add_argument("--interval",type=str,required=False,help="Indiquer l'interval de temps (1min, 5min, 15min, 30min, 60min)")
    parser_harvest.add_argument("--csv",action='store_true',required=False,help="Indiquer si un fichier csv doit être généré")

    parser_analyses = subparsers.add_parser('analyses', help='Exécute la commande analyses')
    parser_analyses.add_argument("--symbol",type=str,required=True,help="Indiquer le symbole à analyser")
    parser_analyses.add_argument("--start",type=str,required=False,help="Indiquer l'annee à analyser")
    parser_analyses.add_argument("--end",type=str,required=False,help="Indiquer l'annee à analyser")

    

    args = parser.parse_args()

    if args.commande == 'delete':
        cleanTrainingData(args.symbol, args.year)
    if args.commande == 'harvestYear':
        if args.csv:
            harvestYear(args.year,args.symbol, args.interval, True)
        else:
            harvestYear(args.year, args.symbol, args.interval)
    elif args.commande == 'analyses':
        analyse(args.symbol,args.start,args.end, 32)
if __name__ == "__main__":
    #getMaxMinData(60,"NVDA", "2024-03-15")
    main()