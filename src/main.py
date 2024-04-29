
from fastapi import Depends, FastAPI, HTTPException
from .database import get_db
from src.settings import get_settings
from src.auth.router import router as auth
from src.user.router import router as user
from src.moex.router import router as moex
from sqlalchemy.orm import Session

def get_app() -> FastAPI:
    application = FastAPI(title=get_settings().project_name)
    application.include_router(auth)
    application.include_router(user)
    application.include_router(moex)
    return application
app = get_app()





