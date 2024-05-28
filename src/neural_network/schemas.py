from datetime import datetime
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel


class Loss(str,Enum):
    mse = "mse"
   
    
class Optimizer(str,Enum):
    adam='adam'

class Period(BaseModel):
    start: datetime
    end: datetime

class Params(BaseModel):
    treningAsset:int
    treningDate: Period
    testingAsset: int
    testingDate: Period
    sequenceLength: int
    normalise: bool
    numberEpochs: int
    batchSize: int
    loss: Loss
    optimizer: Optimizer

class LstmClass(BaseModel):
    type: str='lstm'
    neurons: int
    input_timesteps: Optional[int]= None
    input_dim: Optional[int]= None
    return_seq: bool

class DropoutClass(BaseModel):
    type:str='dropout'
    rate:float

class DenseClass(BaseModel):
    type:str='dense'
    neurons: int
    activation:str='linear'

class Settings(BaseModel):
    params:Params
    layers:List[Union[LstmClass, DropoutClass, DenseClass]]
