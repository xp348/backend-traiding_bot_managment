from datetime import datetime, timedelta
from typing import Annotated, Union

import requests
from typing_extensions import Doc




from typing import List, Dict, Any

from .MOEX_ISS_schemas import ColumnsName, History, Metadata, Quotes






def get_quotes(
        start_date:Annotated[
            datetime,
            Doc("Start date for historical data (YYYY-MM-DD)"),
        ],
        end_date:Annotated[
            datetime,
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


    all_data = []
    step_days=100
    current_start_date = start_date
    current_end_date = start_date+timedelta(days=step_days)
    if current_end_date > end_date:
        current_end_date=end_date
    history_data = History(
                metadata=Metadata(),
                columns=[],
                data=[]
            )
            
    while True:
        params = {
            'start': 0,    # Начальный индекс результата
            'iss.only': 'history',
            'history.columns': 'TRADEDATE,OPEN,CLOSE,HIGH,LOW,VOLUME',  # Поля, которые вы хотите получить
            'from': current_start_date.strftime('%Y-%m-%d'),  # Начальная дата периода
            'till': current_end_date.strftime('%Y-%m-%d'),    # Конечная дата периода

            }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            json_data = response.json()
            history_data = History(
                metadata=Metadata(**json_data['history']['metadata']),
                columns=[ColumnsName(column) for column in json_data['history']['columns']],
                data=[]
            )
            all_data.extend(json_data['history']['data'])
            if current_end_date == end_date:
                break
            current_start_date = current_end_date + timedelta(days=1)
            current_end_date = current_start_date+timedelta(days=step_days)
            if current_end_date > end_date:
                current_end_date=end_date
        else:
           return False
        
        
    history_data.data=all_data
    quotes = Quotes(history=history_data)
    return quotes
