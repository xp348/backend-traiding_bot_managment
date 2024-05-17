


from fastapi import APIRouter, HTTPException

from core.NeuralNetwork.NeuralNetwork import NeuralNetwork
from core.requests.MOEX_ISS import get_quotes
from core.requests.MOEX_ISS_schemas import Quotes
from .service import data_conversion

from .schemas import  Settings




data = {
    "params": {
        "treningAsset": 0,
        "treningDate": {
            "start": "2020-03-25",
            "end": "2023-04-02"
        },
        "testingAsset": 0,
        "testingDate": {
            "start": "2020-03-25",
            "end": "2023-04-02"
        },
        "sequenceLength": 0,
        "normalise": True,
        "numberEpochs": 0,
        "batchSize": 0,
        "loss": "mse",
        "optimizer": "adam"
    },
    "layers": []
}

settings_obj = Settings(**data)



router = APIRouter( prefix="/neural-network", tags=["NeuralNetwork"])

@router.post("/bot" )
async def patch_bot(settings: Settings):
    security: str = 'SBER'
    settings = Settings(**data)
    result:Quotes| bool=get_quotes(settings.params.treningDate.start,settings.params.treningDate.end,security)
    if result==False:
        raise HTTPException(status_code=400, detail="Ошибка парсинга данных для обучения")
    treningDate=data_conversion(result.history.data) 
    
    result:Quotes| bool=get_quotes(settings.params.testingDate.start,settings.params.testingDate.end,security)
    if result==False:
        raise HTTPException(status_code=400, detail="Ошибка парсинга тестовых данных")
    testingDate= data_conversion(result.history.data) 

    neuralNetwork=NeuralNetwork()

    neuralNetwork.get_data(treningDate,testingDate)
    loss,val_loss= neuralNetwork.training()
    return neuralNetwork.get_forecast(treningDate,testingDate)
    # return  {
    #     'treningDate':treningDate,
    #     'testingDate':testingDate,
    #     'averageError':{
    #         'loss':loss,
    #         'val_loss':val_loss
    #     }
    # }