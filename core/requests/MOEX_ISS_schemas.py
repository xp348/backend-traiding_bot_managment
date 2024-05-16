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
    bytes:Optional[int]=None
    max_size:Optional[int]=None

class ColumnsName(Enum):
    TRADEDATE = "TRADEDATE"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    HIGH = "HIGH"
    LOW = "LOW"
    VOLUME = "VOLUME"
   
class Metadata (BaseModel):
    TRADEDATE :Optional[MetadataItem]=None
    OPEN :Optional[MetadataItem]=None
    CLOSE:Optional[MetadataItem]=None
    HIGH :Optional[MetadataItem]=None
    LOW :Optional[MetadataItem]=None
    VOLUME :Optional[MetadataItem]=None

class History(BaseModel):
    metadata:Metadata
    columns: List[ColumnsName]=[]
    data: List[List[Union[str, float, int]]]=[]

class Quotes(BaseModel):
    history: History