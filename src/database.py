
from datetime import datetime
from src.settings import get_settings
from sqlalchemy import TIMESTAMP, Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import  create_async_engine,AsyncSession
from sqlalchemy.orm import DeclarativeBase,sessionmaker



async_engine = create_async_engine(
    get_settings().async_database_url,
    #future=True,
    # echo=True,
    )
async_session = sessionmaker(
   bind= async_engine,
   class_=AsyncSession
    # autocommit=False, autoflush=False, 
    )


# Base = declarative_base()
class Base(DeclarativeBase):
    pass


# Dependency
async  def get_db():
    db = async_session()
    try:
        yield db
    finally:
        db.close()


#Models
class User(Base):
    __tablename__ = "user"

    id=Column(Integer,primary_key=True)
    login= Column( String, nullable=False)
    pasword=Column( String, nullable=False)
    registered_at= Column( TIMESTAMP,nullable=False,default=datetime.utcnow )
    check= Column( Float,nullable=False,default=0)
