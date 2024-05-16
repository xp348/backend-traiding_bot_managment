from typing import Annotated, Union

import requests
from typing_extensions import Doc




from typing import List, Dict, Any

from .MOEX_ISS_schemas import ColumnsName, History, Metadata, Quotes






def get_quotes(
        start_date:Annotated[
            str,
            Doc("Start date for historical data (YYYY-MM-DD)"),
        ],
        end_date:Annotated[
            str,
            Doc("End date for historical data (YYYY-MM-DD)"),
        ],
        security=Annotated[
            str,
            Doc(
                """
                """
            ),
        ],
        board:str='TQBR',
        market:str='shares',
        engine:str='stock'
        )-> Union[Quotes, bool] :
    r"""
    Получить историю торгов для указанной бумаги на указанном режиме торгов за указанный интервал дат.
    """
    url = f'https://iss.moex.com/iss/history/engines/{engine}/markets/{market}/boards/{board}/securities/{security}.json'
    params = {
        'start': 0,    # Начальный индекс результата
        'iss.only': 'history',
        'history.columns': 'TRADEDATE,OPEN,CLOSE,HIGH,LOW,VOLUME',  # Поля, которые вы хотите получить
        'from': start_date,  # Начальная дата периода
        'till': end_date,    # Конечная дата периода
      
        }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        json_data = response.json()
        history_data = History(
            metadata=Metadata(**json_data['history']['metadata']),
            columns=[ColumnsName(column) for column in json_data['history']['columns']],
            data=json_data['history']['data']
        )
        quotes = Quotes(history=history_data)
        return quotes
    else:
       False
