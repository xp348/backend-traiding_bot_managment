from typing import Optional

from pydantic import BaseModel


class OkOut(BaseModel):
    """
    Модель для ответа с флагом "ok" и деталями.

    Attributes:
        - ok: флаг успешности операции.
        - details: дополнительные детали ответа.
    """
    ok: bool = True
    details: list = []


class PaginationOut(BaseModel):
    """
    Модель для ответа с информацией о пагинации.

    Attributes:
        - length: общее количество элементов.
        - limit: максимальное количество элементов, выводимых на страницу.
        - offset: номер первого элемента на странице.
    """
    length: Optional[int]
    limit: Optional[int]
    offset: Optional[int]

class ErrorResponse(BaseModel):
    """
    Модель ErrorResponse используется для представления ошибок API.
    Она включает следующие поля:
    - error: строка ошибки
    - detail: дополнительная информация об ошибке (опционально)
    - message: сообщение с подробным описанием ошибки (опционально)
    """
    error: str
    detail: Optional[str]
    message: str