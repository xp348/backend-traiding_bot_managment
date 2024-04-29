
from datetime import  timedelta
from typing import Annotated
from typing_extensions import Doc

from fastapi import APIRouter, Depends, HTTPException, Query,  status
import requests
from fastapi.security import  OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import ResponseHistory
# from src import database
# from .service import
from .MOEX_ISS import history


# from .schemas import  User







router = APIRouter( prefix="/moex", tags=["MOEX"])


@router.get("/quotes",response_model=ResponseHistory)
async def read_user(
    security: str = 'SBER',
    start_date: str = Query('2020-03-25', description="Start date for historical data (YYYY-MM-DD)"),
    end_date: str = Query('2023-04-02', description="End date for historical data (YYYY-MM-DD)"),
    
   
):
    return history(start_date,end_date,security)
    
   
