
from datetime import datetime
from src.settings import get_settings
from typing import Optional,Annotated
from sqlalchemy import TIMESTAMP, Column, Float, Integer, String ,ForeignKey,text,func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import  create_async_engine,AsyncSession
from sqlalchemy.orm import DeclarativeBase,sessionmaker,Mapped,mapped_column, relationship
import enum


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

# str_256=Annotated[str,256]
class Base(DeclarativeBase):
    # type_annotation_map ={
    #     str_256:String(256)
    # }
    pass


# Dependency
async  def get_db():
    db = async_session()
    try:
        yield db
    finally:
        db.close()



#Enums
class ModelScale(enum.Enum):
    one_minute='one_minute'
    five_minutes='five_minutes'
    ten_minutes='ten_minutes'
    fifteen_minutes='fifteen_minutes'
    thirty_minutes='thirty_minutes'
    hour='hour'
    day='day'
    week='week'
    month='month'

class TransactionType(enum.Enum):
    buy='buy'
    sell='sell'
#Types
intpk=Annotated[int,mapped_column(primary_key=True)]
created_at=Annotated[datetime,mapped_column(server_default=text("TIMEZONE('utc', now())"))]
updated_at=Annotated[datetime,mapped_column(server_default=text("TIMEZONE('utc', now())"),onupdate=datetime.utcnow)]

#Models
class User(Base):
    __tablename__ = "user"

    # id=Column(Integer,primary_key=True)
    # login= Column( String, nullable=False)
    # pasword=Column( String, nullable=False)
    # registered_at= Column( TIMESTAMP,nullable=False,default=datetime.utcnow )
    # check= Column( Float,nullable=False,default=0)

    id:Mapped[intpk]
    login: Mapped[str]
    pasword: Mapped[str]
    registered_at: Mapped[created_at] 
    check: Mapped[float]=mapped_column(default=0)

    _models:  Mapped[list["Model"]]=relationship("_user")
    _transactions: Mapped[list["Transaction"]] =relationship(back_populates="_user")
    _assets: Mapped[list["Asset"]]=relationship(back_populates="_users",secondary="user_asset")

class UserAsset (Base): 
    '''промежуточная таблица many to many'''
    __tablename__ = "user_asset"

    # user_id=Column(Integer,ForeignKey('user.id',ondelete="CASCADE"),primary_key=True, nullable=False)
    # assets_id=Column(Integer,ForeignKey('asset.id',ondelete="CASCADE"),primary_key=True, nullable=False)

    user_id: Mapped[int] = mapped_column(ForeignKey('user.id',ondelete="CASCADE"),primary_key=True)
    assets_id: Mapped[int]= mapped_column(ForeignKey('asset.id',ondelete="CASCADE"),primary_key=True)

class Asset (Base):
    __tablename__ = "asset"
    id :Mapped[intpk]
    name :Mapped[str]
    asset_type:Mapped[str]#?
    created_at:Mapped[created_at]
    created_assets_at:Mapped[datetime]

    # id =Column(Integer,primary_key=True)
    # name =Column(String, nullable=False)
    # asset_type=Column(String, nullable=False)#?
    # created_at=Column( TIMESTAMP, nullable=False,default=datetime.utcnow )
    # created_assets_at=Column( TIMESTAMP, nullable=False )

    _users: Mapped[list["User"]]=relationship(back_populates="_assets",secondary="user_asset")
    _models:  Mapped[list["Model"]]=relationship(back_populates="_asset")
    _transactions: Mapped[list["Transaction"]] =relationship(back_populates='_asset')
   


class Model(Base):
    __tablename__ = "model"
    id:Mapped[intpk]
    name:Mapped[str]
    user_id :Mapped[int]=mapped_column(ForeignKey('user.id',ondelete="CASCADE"))
    asset_id :Mapped[int]=mapped_column(ForeignKey('asset.id',ondelete="CASCADE"))
    model_scale :Mapped[ModelScale]
    created_at :Mapped[created_at]
    architecture :Mapped[str]#json

    # id=Column(Integer,primary_key=True)
    # name=Column(String, nullable=False)
    # user_id =Column(Integer,ForeignKey('user.id',ondelete="CASCADE"))
    # asset_id =Column(Integer,ForeignKey('asset.id',ondelete="CASCADE"))
    # model_scale =Column(String, nullable=False)#ModelScale
    # created_at =Column( TIMESTAMP, nullable=False,default=datetime.utcnow )
    # architecture =Column(String, nullable=False)#json

    _user: Mapped[list["User"]] =relationship(back_populates="_models")
    _asset: Mapped[list["Asset"]] =relationship(back_populates="_models")

class Transaction(Base): 
    __tablename__ = "transaction"
    id:Mapped[intpk]
    user_id :Mapped[int]=mapped_column(ForeignKey('user.id',ondelete="CASCADE"))
    asset_id :Mapped[int]=mapped_column(ForeignKey('asset.id',ondelete="CASCADE"))
    transaction_type:Mapped[TransactionType]
    quantity:Mapped[int]
    price:Mapped[int]
    balance:Mapped[int]
    created_at:Mapped[created_at]

    # id=Column(Integer,primary_key=True)
    # user_id =Column(Integer,ForeignKey('user.id',ondelete="CASCADE"))
    # asset_id =Column(Integer,ForeignKey('asset.id',ondelete="CASCADE"))
    # transaction_type=Column(String, nullable=False)#TransactionType
    # quantity=Column(Integer, nullable=False)
    # price=Column(Integer, nullable=False)
    # balance=Column(Integer, nullable=False)
    # created_at =Column( TIMESTAMP, nullable=False,default=datetime.utcnow )

    _user: Mapped[list["User"]] =relationship(back_populates="_transactions")
    _asset: Mapped[list["Asset"]] =relationship(back_populates="_transactions")
