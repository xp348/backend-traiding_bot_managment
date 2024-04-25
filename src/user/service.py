from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src import database
from . import schemas

async def get_user_by_login(db:AsyncSession, login:str):
    result = await db.execute(select(database.User).filter(database.User.login == login))
    return result.scalar_one_or_none()

async def get_user_by_id(db:AsyncSession, id:int):
    result = await db.execute(select(database.User).filter(database.User.id == id))
    return result.scalar_one_or_none()

async def create_user(db: AsyncSession, user: schemas.UserCreate):
    fake_hashed_password = user.pasword + "notreallyhashed"
    db_user = database.User(login=user.login, pasword=fake_hashed_password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

async def remove_user(db: AsyncSession, id: int):
    db_user =await db.execute(select(database.User).filter(database.User.id == id))
    user = await db_user.scalar_one_or_none()
    if user:
        db.delete(user)  
        await db.commit() 
        return True
    return False

async def get_users(db: AsyncSession,  limit: int, offset:int):
    result = await db.execute(select(database.User).limit(limit).offset(offset))
    return result.scalars().all()