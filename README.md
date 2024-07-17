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
