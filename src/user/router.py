from typing import Annotated
from fastapi import APIRouter, Depends,HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.auth.service import get_current_active_user
from src import database, schemas as sch
from src.schemas import OkOut,ErrorResponse
from .service import get_user_by_login,create_user,get_user_by_id,get_users,remove_user
from .schemas import User,UserCreate

router = APIRouter( prefix="/user", tags=["User"])




@router.post("/create", response_model=User)
async def post_user(user: UserCreate,  db: AsyncSession = Depends(database.get_db)):
    db_user = await get_user_by_login(db, user.login)
    if db_user:
        raise HTTPException(status_code=400, detail="Пользователь с таким логином уже существует")
    return await create_user(db=db, user=user)

@router.get("/by/{user_id}", response_model=User)
async def read_user(user_id: int, db: AsyncSession = Depends(database.get_db)):
    db_user = await get_user_by_id(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="Пользователь с таким id не найден")
    return db_user

@router.get("/list", response_model=list[User])
async def read_users(offset: int = 0, limit: int = 100, db: AsyncSession = Depends(database.get_db)):
    return await get_users(db, limit,offset) 


@router.delete("/delete/{user_id}", response_model=OkOut,
                summary="не работает",
    description="Получить клиентов по тэгам",
    responses={
        200: {"description": "Клиент успешно удален"},
        422: {
            "description": "ййй",
            "model": ErrorResponse,
        },
    },)
async def delete_user(user_id: int, db: AsyncSession = Depends(database.get_db)):
    """не работатет"""
    result = sch.OkOut()
    result.ok = await remove_user(db, user_id)
    return result
   