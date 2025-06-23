from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from pathlib import Path

from .abstract_factory import AbstractFactory, AbstractProduct
from .confs import google_confs, safehouse


class GoogleProduct(AbstractProduct):
    """
    Producto utilizado para manejo de archivo en Drive.
    """

    def __init__(self, type: str='drive'):
        super().__init__()
        self.scopes = google_confs['scopes']
        self.creds = self._get_credentials()
        self._connect()

    def _get_credentials(self) -> Credentials:
        """
        Carga las credenciales desde token.json, las refresca si han expirado
        o inicia un nuevo flujo OAuth interactivo. Al final guarda siempre
        el token actualizado en disco.
        """
        creds = None
        token_path = safehouse / google_confs.get('token_filename')
        client_path = safehouse / google_confs.get('client_filename')

        # Intentar cargar credenciales previas
        if Path(token_path).is_file():
            creds = Credentials.from_authorized_user_file(
                token_path, self.scopes
            )

        # Si no hay credenciales válidas, refrescar o solicitar login
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                # Refresca automáticamente usando el refresh token
                creds.refresh(Request())
            else:
                # Primer login: lanza el flujo de OAuth en local server
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_path, self.scopes
                )
                creds = flow.run_local_server(port=0)

            # Guardar (o actualizar) token.json con access+refresh tokens
            with open(token_path, "w") as token_file:
                token_file.write(creds.to_json())
        return creds

    def _connect(self, type='drive'):
        self.type = type.lower()
        if self.type == 'spreadsheet':
            self.type = type
            self.service = build('sheets', 'v4', credentials=self.creds)
        else:
            self.type = 'drive'
            self.service = self.service = build('drive', 'v3', credentials=self.creds)


class GoogleFactory(AbstractFactory):
    def run(self, type: str='drive') -> GoogleProduct:
        return GoogleProduct(type)
