
from datetime import datetime
from keras.optimizers import Adam #Оптимизатор
from keras.models import Sequential, Model #Два варианты моделей
from keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, LSTM #Стандартные слои
from keras.preprocessing.sequence import TimeseriesGenerator # для генерации выборки временных рядов
import tensorflow as tf

class Data():
    def __init__(self,Date:list[datetime],Open:list[float],Close:list[float],High:list[float],Low:list[float],Volume:list[int]):
       self.Date   = Date
       self.Open   = Open
       self.Close  = Close
       self.High   = High
       self.Low    = Low
       self.Volume = Volume


class NeuralNetwork():
    def __init__(self):
        self.data_train=1
        self.data_test =1
        self.window_size = 30 # 
        self.batch_size = 20


    def get_data(self,data_train:Data,data_test:Data):
        def data_formalization(data:Data):
            x =  data.Date
            y= [[x] for x in data.Close]
            return TimeseriesGenerator(x, y,           #В качестве параметров наши выборки
                               length=self.window_size, stride=1, #Для каждой точки (из промежутка длины xLen)
                               batch_size=self.batch_size)                #Размер batch, который будем скармливать модели
            
        
        self.x_train, self.y_train = data_formalization(data_train)
        self.x_test , self.y_test   = data_formalization(data_test)
        return 