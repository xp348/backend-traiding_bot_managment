
from fastapi import APIRouter,Query
from .schemas import ResponseHistory
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
    
   