
from datetime import  datetime, timedelta
from typing import Annotated, Any, List
from typing_extensions import Doc

from fastapi import APIRouter, Depends, HTTPException, Query,  status
import requests
from fastapi.security import  OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from core.requests.MOEX_ISS_schemas import History, Quotes
# from src import database
# from .service import
from core.requests.MOEX_ISS import get_quotes


# from .schemas import  User







router = APIRouter( prefix="/moex", tags=["MOEX"])


@router.get("/quotes",response_model=History )
async def read_user(
    security: str = 'SBER',
    start_date: datetime = Query('2020-03-25', description="Start date for historical data (YYYY-MM-DD)"),
    end_date: datetime = Query('2023-04-02', description="End date for historical data (YYYY-MM-DD)"),
    
   
)->History :
    result:Quotes| bool=get_quotes(start_date,end_date,security)
    if result==False:
        raise HTTPException(status_code=400, detail="---")
    return result.history
