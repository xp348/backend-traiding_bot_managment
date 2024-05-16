
from fastapi import FastAPI

from src.settings import get_settings
from src.auth.router import router as auth
from src.user.router import router as user
from src.quotes.router import router as moex
from src.neural_network.router import router as neural_network



def get_app() -> FastAPI:
    application = FastAPI(title=get_settings().project_name)
    application.include_router(auth)
    application.include_router(user)
    application.include_router(moex)
    application.include_router(neural_network)
    return application

app = get_app()

# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="Custom title",
#         version="2.5.0",
#         summary="This is a very custom OpenAPI schema",
#         description="Here's a longer description of the custom **OpenAPI** schema",
#         routes=app.routes,
#     )
#     openapi_schema["info"]["x-logo"] = {
#         "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
#     }
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

# app.openapi = custom_openapi

# Uncaught ReferenceError: SwaggerUIBundle is not defined     fastapi




