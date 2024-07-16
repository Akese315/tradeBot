import argparse
import sys
from typing import List
from dotenv import load_dotenv
import requests
import mysql.connector
import os
from time import sleep
import pandas as pd
from datetime import datetime, timedelta, date
load_dotenv()

ALPHA_VENTAGE_KEY = os.getenv('ALPHA_VENTAGE_KEY_1')
PASSWORD = os.getenv('PSW')
HOST = os.getenv('HOST')
USR = os.getenv('USR')
DATABASE = os.getenv('DATABASE')



cursor = None
conn = None
    
#connection to the database
try:
    print("Connecting to the database : ",USR,'@',HOST,"to", DATABASE,"with :", PASSWORD)
    conn = mysql.connector.connect(host=HOST,user=USR,password=PASSWORD,database=DATABASE)
    
except Exception as e:
    print(e)

cursor = conn.cursor()

def getMaxMinData(period:int, symbol:str, current_date:str) -> tuple[float,float]:
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

def getLocalTrainingData(symbol:str) -> List:
    SQL_SELECT = "SELECT time, closePrice, openPrice, lowPrice, highPrice FROM training_data_hour WHERE symbol='"+symbol+"' ORDER BY time ASC;"
    cursor.execute(SQL_SELECT)
    rows = cursor.fetchall()
    return rows

def cleanTrainingData(symbol:str, year:str):
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
    

    args = parser.parse_args()

    if args.commande == 'delete':
        cleanTrainingData(args.symbol, args.year)
    elif args.commande == 'harvestYear':
        if args.csv:
            harvestYear(args.year,args.symbol, args.interval, True)
        else:
            harvestYear(args.year, args.symbol, args.interval)

if __name__ == "__main__":
    #getMaxMinData(60,"NVDA", "2024-03-15")
    main()