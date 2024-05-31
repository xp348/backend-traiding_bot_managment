


import json
from fastapi import APIRouter, HTTPException
from sklearn.preprocessing import MinMaxScaler


from core.NeuralNetwork.NeuralNetwork import DataLoader, NeuralNetworkModel
from core.requests.MOEX_ISS import get_quotes
from core.requests.MOEX_ISS_schemas import Quotes
from .service import data_conversion

from .schemas import  Settings
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator # для генерации выборки временных рядов
from keras.optimizers import Adam #Оптимизатор
from keras.models import Sequential #Два варианты моделей
from keras.layers import concatenate, Input, Dense,  LSTM #Стандартные слои

data = {
    "params": {
        "treningAsset": 0,
        "treningDate": {
            # "start": "2020-03-25",
            "start": "2000-01-05",
            "end": "2022-01-01"
        },
        "testingAsset": 0,
        "testingDate": {
            "start": "2022-02-01",
            "end": "2023-01-01"
        },
        "sequenceLength": 50,
        "normalise": True,
        "numberEpochs": 10,
        "batchSize": 32,
        "loss": "mse",
        "optimizer": "adam"
    },
    "layers": [
        {
        "type": "lstm",
        "neurons": 100,
        "input_timesteps": 49,
        "input_dim": 2,
        "return_seq": True
      },
      {
        "type": "dropout",
        "rate": 0.2
      },
      {
        "type": "lstm",
        "neurons": 100,
        "return_seq": True
      },
      {
        "type": "lstm",
        "neurons": 100,
        "return_seq": True
      },
      {
        "type": "dropout",
        "rate": 0.2
      },
      {
        "type": "dense",
        "neurons": 1,
        "activation": "linear"
      }
    ]
}





router = APIRouter( prefix="/neural-network", tags=["NeuralNetwork"])

@router.post("/bot" )
async def patch_bot(settings: Settings):
    security: str = 'SBER'
    timeframe = "D1"
    settings = Settings(**data)
    treningResult:Quotes| bool=get_quotes(settings.params.treningDate.start,settings.params.treningDate.end,security)
    if treningResult==False:
        raise HTTPException(status_code=400, detail="Ошибка парсинга данных для обучения")
    treningDate=data_conversion(treningResult.history.data) 
    
    testingResult:Quotes| bool=get_quotes(settings.params.testingDate.start,settings.params.testingDate.end,security)
    if testingResult==False:
        raise HTTPException(status_code=400, detail="Ошибка парсинга тестовых данных")
    testingDate= data_conversion(testingResult.history.data) 
    ####
    # return {
    #     'treningResult':treningResult,
    #     'testingResult':testingResult
    #     'treningDate':treningDate,
    #     'testingDate':testingDate,
    #     'treningDate_len':len(treningDate.Close),
    #     'testingDate_len':len(testingDate.Close),
    # }
    dataLoader = DataLoader()
    dataLoader.data_train = np.array([np.array([close,volume]) for close,volume in zip(treningDate.Close,treningDate.Volume)])
    dataLoader.data_test = np.array([np.array([close,volume]) for close,volume in zip(testingDate.Close,testingDate.Volume)])
    dataLoader.len_train =len(dataLoader.data_train)
    dataLoader.len_test = len(dataLoader.data_test )
    # return {
    #     'data_train':dataLoader.data_train,
    #     'data_test':dataLoader.data_test,
    #     'len_train':dataLoader.len_train,
    #     'len_test':dataLoader.len_test,
    # }
    neuralNetworkModel = NeuralNetworkModel()
    neuralNetworkModel.build_model(settings.params.loss,settings.params.optimizer,settings.layers)

    x, y = dataLoader.get_train_data(
        seq_len=settings.params.sequenceLength,
        normalise=settings.params.normalise #False
    )
   
    neuralNetworkModel.train(
      x,
      y,
      epochs = settings.params.numberEpochs,
      batch_size = settings.params.batchSize,
      timeframe=timeframe
    )
    # return {
    #     'f':y.tolist()
    # }
    x_test, y_test = dataLoader.get_test_data(
        seq_len=settings.params.sequenceLength,
        normalise=settings.params.normalise #False
    )
    # return {
    #     'f':1
    # }
    predictions = neuralNetworkModel.predict_point_by_point(x_test)
    y_test=y_test.tolist()
  
    # return {'predictions':predictions.tolist(),'true_data':y_test}
    return {'true_data':1}
    # return  {
    #     'treningDate':treningDate,
    #     'testingDate':testingDate,
    #     'averageError':{
    #         'loss':loss,
    #         'val_loss':val_loss
    #     }
    # }

def getPred(currModel, xVal, yVal, yScaler):
  # Предсказываем ответ сети по проверочной выборке
  # И возвращаем исходны масштаб данных, до нормализации
  predVal = yScaler.inverse_transform(currModel.predict(xVal))
  yValUnscaled = yScaler.inverse_transform(yVal)    
  return (predVal, yValUnscaled)

@router.post("/bot2" )
async def patch_bot(settings: Settings):
    security: str = 'SBER'
  
    # settings = Settings(**data)
    treningResult:Quotes| bool=get_quotes(settings.params.treningDate.start,settings.params.treningDate.end,security)
    if treningResult==False:
        raise HTTPException(status_code=400, detail="Ошибка парсинга данных для обучения")
    xTrain=np.array(treningResult.history.data)[:,1:] #убераем даты    
    testingResult:Quotes| bool=get_quotes(settings.params.testingDate.start,settings.params.testingDate.end,security)
    if testingResult==False:
        raise HTTPException(status_code=400, detail="Ошибка парсинга тестовых данных")
    xTest= np.array(testingResult.history.data)[:,1:] 

    # xLen = 45# 300                      #Анализируем по 300 прошедшим точкам
    #Масштабируем данные (отдельно для X и Y), чтобы их легче было скормить сетке
    xScaler = MinMaxScaler()
    xScaler.fit(xTrain)
    xTrain = xScaler.transform(xTrain)
    xTest = xScaler.transform(xTest)
    
    #Делаем reshape,т.к. у нас только один столбец по одному значению
    yTrain,yTest = np.reshape(xTrain[:,3],(-1,1)), np.reshape(xTest[:,3],(-1,1))
    yScaler = MinMaxScaler()
    yScaler.fit(yTrain)
    yTrain = yScaler.transform(yTrain)
    yTest = yScaler.transform(yTest)
   
    #Создаем генератор для обучения
    trainDataGen = TimeseriesGenerator(xTrain, yTrain,           #В качестве параметров наши выборки
                                   length=settings.params.sequenceLength, stride=1, #Для каждой точки (из промежутка длины xLen)
                                   batch_size=settings.params.batchSize)                #Размер batch, который будем скармливать модели
    
    #Создаем аналогичный генератор для валидации при обучении
    testDataGen = TimeseriesGenerator(xTest, yTest,
                                   length=settings.params.sequenceLength, stride=1,
                                   batch_size=settings.params.batchSize)
    

        #Создадим генератор проверочной выборки, из которой потом вытащим xVal, yVal для проверки
    DataGen = TimeseriesGenerator(xTest, yTest,
                                   length=settings.params.sequenceLength, sampling_rate=1,
                                   batch_size=len(xTest)) #размер batch будет равен длине нашей выборки
    xVal = []
    yVal = []
    for i in DataGen:
      xVal.append(i[0])
      yVal.append(i[1])

    xVal = np.array(xVal)
    yVal = np.array(yVal)


    modelL = Sequential()
    modelL.add(LSTM(5, input_shape = (settings.params.sequenceLength, 5)))
    modelL.add(Dense(10, activation="linear"))
    modelL.add(Dense(1, activation="linear"))

    modelL.compile(loss="mse", optimizer=Adam(lr=1e-5))

    history = modelL.fit_generator(trainDataGen,
                        epochs=settings.params.numberEpochs,
                        verbose=1,
                        validation_data=testDataGen)
    

    #Прогнозируем данные текущей сетью
    currModel = modelL
    (predVal, yValUnscaled) = getPred(currModel, xVal[0], yVal[0], yScaler)

    loss=np.where(np.isnan(history.history['loss']), None, history.history['loss'])
    val_loss=np.where(np.isnan(history.history['val_loss']), None, history.history['val_loss'])
    predVal=np.where(np.isnan(predVal), None, predVal)
    yValUnscaled=np.where(np.isnan(yValUnscaled), None, yValUnscaled)
    # predVal=np.where(np.isinf(predVal), None, predVal)

    return {
        # 'xTrain':np.where(np.isnan(xTrain), None, xTrain).tolist(),
        # 'yTrain':np.where(np.isnan(yTrain), None, yTrain).tolist(),
        # 'xTest':np.where(np.isnan(xTest), None, xTest).tolist(),
        # 'yTest':np.where(np.isnan(yTest), None, yTest).tolist(),
        'loss':loss.tolist(),
         'val_loss':val_loss.tolist(),
         'dataTrain':treningResult.history.data,
        'dataTest':testingResult.history.data,
       'predVal':predVal.tolist(),
        'yValUnscaled':yValUnscaled.tolist()
    }
   