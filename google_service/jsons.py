from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from json import loads
import logging
import pandas as pd
from typing import Optional

from .base import GoogleFactory, GoogleProduct


class JsonProduct(GoogleProduct):
    def __init__(self) -> None:
        super().__init__('drive')
        self.type = 'json'
        self._initialize_service()
    
    def run(self) -> any:
        return 'Service'

    def _initialize_service(self):
        self.service = build('drive', 'v3', credentials=self.creds)
    
    def read_json(self, file_id: str):
        try:
            # Inicializar servicio
            self._initialize_service()
            file_content = (
                self.service.files()
                .get_media(fileId=file_id).execute()
            )
            json_data = loads(file_content.decode("utf-8"))
            return json_data

        except HttpError as error:
            # TODO(developer) - Handle errors from drive API.
            print(f"An error occurred: {error}")

        except Exception:
            logging.exception(
                "Unexpected error reading %s", file_id
            )
            raise


class JsonFactory(GoogleFactory):
    def run(self) -> JsonProduct:
        return JsonProduct()