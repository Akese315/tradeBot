CREATE TABLE stocks(
    id INT PRIMARY KEY AUTO_INCREMENT,
    time TIMESTAMP NOT NULL DEFAULT NOW(),
    currentPrice FLOAT,
    percentChange FLOAT,
    changePrice FLOAT,
    highPriceDay FLOAT,
    lowPriceDay FLOAT,
    sma FLOAT,
    ema FLOAT,
    osc FLOAT,
    osc_ema FLOAT
);

CREATE DATABASE TRADEBOT;