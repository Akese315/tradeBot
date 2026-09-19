# **Trading bot:**

This is a TradingBot powered by AI (PyTorch)

## Command line :

Data Manager :

```bash
python3 dataManager 	delete --symbol <str> --year (optional) <str>
			harvestYear --symbol <str> --year <str> --csv --interval <str> (1min, 5min, 15min, 30min, 60min) 
```

Bot :

```bash
python3 bot.py	start --symbol <str> --nogui (optional) --server (optionnal)
		train --symbol <str>
```

> By default the bot start with GUI

## Database :

You need to create a database with the exact same features.

See the database.sql

## Dataframe

Please create a dataframe folder

## Model

Please create a model folder

version 1 is 300 hidden size
version 2 and 3 are 600 hidden size

version 7 is 600 hidden size with 8 inputs

version 7 is 600 hidden size with 8 inputs

version 13 is 600 siwe with 7 inputs

version 14 is 300 hidden size with 7 inputs

version 15 is 500 hidden size with 7 inputs

version 16 and 17, 600 hidden size with 7 inputs
