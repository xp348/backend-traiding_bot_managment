from datetime import datetime
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    id:int
    login:str
    registered_at:datetime
    check:float

class UserInDB(User):
    pasword: str
