
from datetime import datetime
from keras.optimizers import Adam #Оптимизатор
from keras.models import Sequential, Model #Два варианты моделей
from keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, LSTM #Стандартные слои
from keras.preprocessing.sequence import TimeseriesGenerator # для генерации выборки временных рядов
import tensorflow 
from sklearn.preprocessing import MinMaxScaler
class Data():
    def __init__(self,Date:list[datetime]=[],Open:list[float]=[],Close:list[float]=[],High:list[float]=[],Low:list[float]=[],Volume:list[int]=[]):
       self.Date   = Date
       self.Open   = Open
       self.Close  = Close
       self.High   = High
       self.Low    = Low
       self.Volume = Volume


class NeuralNetwork():
    def __init__(self):
        self.data_train:TimeseriesGenerator
        self.data_test :TimeseriesGenerator
        self.window_size = 30 # 
        self.batch_size = 20
        self.stride=1
        self.model = Sequential()


    def get_data(self,data_train:Data,data_test:Data):

        def data_formalization(data:Data):
            x =  [d.timestamp() for d in data.Date]
            y= [[x] for x in data.Close]
            return TimeseriesGenerator(x, y,           #В качестве параметров наши выборки
                               length=self.window_size, stride=self.stride, #Для каждой точки (из промежутка длины xLen)
                               batch_size=self.batch_size)                #Размер batch, который будем скармливать модели
            
        
        self.data_train = data_formalization(data_train)
        self.data_test   = data_formalization(data_test)
    
    def training(self):
        '''Вернет:
        Средняя абсолютная ошибка на обучающем наборе; 
        Средняя абсолютная ошибка на проверочном наборе;
        '''
        self.model.add(LSTM(5, input_shape = (self.window_size, 1)))
        self.model.add(Dense(10, activation="linear"))
        self.model.add(Dense(1, activation="linear"))

        self.model.compile(loss="mse", optimizer=Adam(lr=1e-5))

        history = self.model.fit_generator(self.data_train, epochs=10, verbose=1,validation_data=self.data_test)

        return history.history['loss'], history.history['val_loss'] #Средняя абсолютная ошибка на обучающем наборе / Средняя абсолютная ошибка на проверочном наборе
    
    def get_forecast(self, data_train:Data,data_test:Data):
        xTest =  [d.timestamp() for d in data_test.Date]
        yTest= [[x] for x in data_test.Close]
        #Создадим генератор проверочной выборки, из которой потом вытащим xVal, yVal для проверки
        DataGen = TimeseriesGenerator(xTest, yTest,
                               length=199, sampling_rate=1,
                               batch_size=len(xTest)) #размер batch будет равен длине нашей выборки
        xVal = []
        yVal = []
        for i in DataGen:
            xVal.append(i[0])
            yVal.append(i[1])

        yTrain= [[x] for x in data_train.Close]
        yScaler = MinMaxScaler()
        yScaler.fit(yTrain)

        predVal = yScaler.inverse_transform(self.model.predict(xVal)) #прогноз
        yValUnscaled = yScaler.inverse_transform(yVal) #базовый ряд

        return 0