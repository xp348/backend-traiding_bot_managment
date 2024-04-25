from pydantic import BaseModel
from datetime import datetime
# import uuid

class User(BaseModel):
    # id: uuid.UUID
    id: int
    login:str
    pasword:str
    registered_at: datetime
    check:float

class UserCreate(BaseModel):
    login:str
    pasword:str