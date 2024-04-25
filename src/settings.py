from functools import lru_cache
from pydantic_settings  import BaseSettings

@lru_cache()
def get_settings():
    return AppSettings()

class AppSettings(BaseSettings):
    project_name:str = 'Management of trading bots'

    DB_HOST:str 
    DB_PORT:str 
    DB_NAME:str 
    DB_USER:str 
    DB_PASS:str 
   
    SECRET_KEY:str
    ALGORITHM:str
    ACCESS_TOKEN_EXPIRE_MINUTES:int

    class Config:
        env_file = ".env"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    
    @property
    def async_database_url(self):
        return self.database_url.replace('postgresql', 'postgresql+asyncpg', 1)
