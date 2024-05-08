
from fastapi import  FastAPI
from src.auth.router import router as auth
from src.moex.router import router as moex


def get_app() -> FastAPI:
    application = FastAPI(title='Management of trading bots')
    application.include_router(auth)
    application.include_router(moex)
    return application
app = get_app()