class Quote:

    def __init__(self,openPriceDay, closePrice, highPrice, lowPrice,time, volume) -> None:
        self.openPrice = openPriceDay
        self.closePrice = closePrice
        self.highPrice = highPrice
        self.lowPrice = lowPrice
        self.time = time
        self.volume = volume

class Dot:
    def __init__(self,time,value) -> None:
        self.time = time
        self.value = value

