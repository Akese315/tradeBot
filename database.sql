CREATE DATABASE TRADEBOT;

CREATE TABLE stocks(
    id INT PRIMARY KEY AUTO_INCREMENT,
    time TIMESTAMP NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    currentPrice FLOAT,
    percentChange FLOAT,
    changePrice FLOAT,
    highPriceDay FLOAT,
    lowPriceDay FLOAT,
    volume FLOAT,
    sma FLOAT,
    ema FLOAT,
    osc FLOAT,
    osc_ema FLOAT
);

CREATE TABLE training_data_hour(
    id INT PRIMARY KEY AUTO_INCREMENT,
    time TIMESTAMP NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    highPrice FLOAT,
    lowPrice FLOAT,
    closePrice FLOAT,
    openPrice FLOAT,
    volume FLOAT
);

