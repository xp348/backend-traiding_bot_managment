
from datetime import datetime
from typing import List, Union
from core.NeuralNetwork.NeuralNetwork import Data
from core.requests.MOEX_ISS_schemas import Quotes


def data_conversion(list:List[List[Union[datetime, float, int]]]):
    data=  Data()
    for item in list:
        data.Date.append(item[0])
        data.Open.append(item[1])
        data.Close.append(item[2])
        data.High.append(item[3])
        data.Low.append(item[4])
        data.Volume.append(item[5])
    return  data