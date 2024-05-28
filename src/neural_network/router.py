


from fastapi import APIRouter, HTTPException


from core.NeuralNetwork.NeuralNetwork import DataLoader, NeuralNetworkModel
from core.requests.MOEX_ISS import get_quotes
from core.requests.MOEX_ISS_schemas import Quotes
from .service import data_conversion

from .schemas import  Settings
import numpy as np



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
    dataLoader = DataLoader()
    dataLoader.data_train = np.array([np.array([close,volume]) for close,volume in zip(treningDate.Close,treningDate.Volume)])
    dataLoader.data_test = np.array([np.array([close,volume]) for close,volume in zip(testingDate.Close,testingDate.Volume)])
    dataLoader.len_train =len(dataLoader.data_train)
    dataLoader.len_test = len(dataLoader.data_test )
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

    x_test, y_test = dataLoader.get_test_data(
        seq_len=settings.params.sequenceLength,
        normalise=settings.params.normalise #False
    )

    predictions = neuralNetworkModel.predict_point_by_point(x_test)
    

    return {'predictions':predictions.tolist(),'true_data':y_test}
    # return  {
    #     'treningDate':treningDate,
    #     'testingDate':testingDate,
    #     'averageError':{
    #         'loss':loss,
    #         'val_loss':val_loss
    #     }
    # }