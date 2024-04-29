from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Union,Optional
from pydantic import BaseModel


# class History(BaseModel):
#     metadata: Dict[str, Any]
#     columns: List[str]
#     data: List[Union[str,float]]

# class ResponseHistory(BaseModel):
#     history: History

class MetadataItem(BaseModel):
    type:str
    bytes:Optional[int]
    max_size:Optional[int]

class ColumnsName(Enum):
    TRADEDATE = "TRADEDATE"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    HIGH = "HIGH"
    LOW = "LOW"
    VOLUME = "VOLUME"
   


class History(BaseModel):
    metadata:Dict[ColumnsName,MetadataItem]
    columns: List[ColumnsName]
    data: List[List[Union[str, float, int]]]

class ResponseHistory(BaseModel):
    history: History