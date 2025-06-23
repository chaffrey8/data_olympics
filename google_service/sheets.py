from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import pandas as pd
from typing import Optional

from .base import GoogleFactory, GoogleProduct


class SheetsProduct(GoogleProduct):
    def __init__(self) -> None:
        super().__init__('spreadsheet')
        self.type = 'spreadsheet'
        self._initialize_service()

    def run(self) -> any:
        return 'Service'

    def _initialize_service(self):
        self.service = build('sheets', 'v4', credentials=self.creds)
        self.__drive_service__ = build('drive', 'v3', credentials=self.creds)

    def read_spreadsheet(
        self,
        file_id: str,
        sheet_name: Optional[str] = None,
        data_range: Optional[str]   = None,
        header_row: Optional[int]   = None,
        max_header_scan: int        = 5
    ) -> pd.DataFrame:
        """
        Lee un Google Sheet y devuelve un DataFrame.
        - creds: objeto Credentials obtenido previamente.
        - spreadsheet_id: ID del archivo en Drive.
        - sheet_name: nombre de pestaña (si data_range no incluye pestaña).
        - data_range: rango A1 completo ("Hoja1!A1:Z100").
        - header_row: fila 1-based con encabezados.
        Si no pasas data_range o header_row, se intentan detectar automáticamente.
        """
        try:
            # Inicializar servicio
            self._initialize_service()

            # Validar que sea un Google Sheet
            mime = (
                self.__drive_service__.files()
                .get(fileId=file_id, fields='mimeType', supportsAllDrives=True)
                .execute().get('mimeType')
            )
            if mime != 'application/vnd.google-apps.spreadsheet':
                raise ValueError(
                    f'El archivo {file_id!r} no es un Google Sheet '
                    f'(mimeType={mime}).'
                )

            # Determinar rango
            if data_range:
                rng = data_range
            else:
                # obtener primer nombre de pestaña si no hay data_range
                if not sheet_name:
                    info = (
                        self.service.spreadsheets()
                        .get(spreadsheetId=file_id)
                        .execute()
                    )
                    sheets = info.get('sheets', [])
                    if not sheets:
                        raise ValueError(
                            f'No se encontraron hojas en {file_id!r}'
                        )
                    sheet_name = sheets[0]['properties']['title']
                # leer toda la hoja
                rng = f'{sheet_name}'

            # Traer valores brutos
            resp = (
                self.service.spreadsheets().values()
                .get(spreadsheetId=file_id,range=rng)
                .execute()
            )
            values = resp.get('values') or []
            if not values:
                logging.info('Hoja vacía: %r', rng)
                return pd.DataFrame()

            # Determinar índice de fila de encabezado (0-based)
            if header_row:
                hr = header_row - 1
            else:
                # buscar la fila con más celdas no vacías en las primeras max_header_scan
                scan = min(len(values), max_header_scan)
                counts = [
                    sum(bool(str(cell).strip()) for cell in values[i])
                    for i in range(scan)
                ]
                hr = int(max(enumerate(counts), key=lambda x: x[1])[0])

            # Separar encabezados y datos
            header = [str(c).strip() for c in values[hr]]
            data_rows = values[hr+1 :]

            # Filtrar sólo las columnas que tengan header no vacío
            valid = [i for i, h in enumerate(header) if h]
            cols  = [header[i] for i in valid]
            data = [
                [row[i] if i < len(row) else '' for i in valid]
                for row in data_rows
            ]
            return pd.DataFrame(data, columns=cols).fillna('')

        except HttpError as error:
            logging.error(
                "Error reading %s (sheet %s): %s",
                file_id,
                sheet_name,
                error
            )
            raise
        except Exception:
            logging.exception(
                "Unexpected error reading %s", file_id
            )
            raise

    def update_spreadsheet(
        self,
        file_id: str,
        sheet_name: str,
        df: pd.DataFrame,
        header_row: int = 1,
    ) -> dict:
        """
        Sobrescribe la hoja `sheet_name` con el contenido de `df`.
        - header_row (1-based): fila donde colocar los nombres de columna.
          Todas las filas anteriores quedan intactas.
        """
        try:
            # Inicializar servicio de Sheets
            self._initialize_service()

            # Obtener tamaño de la hoja para definir el rango de limpieza
            meta = (
                self.service.spreadsheets()
                .get(
                    spreadsheetId=file_id,
                    fields='sheets(properties(title,gridProperties))'
                )
                .execute()
            )
            props = None
            for sh in meta['sheets']:
                p = sh['properties']
                if p['title'] == sheet_name:
                    props = p['gridProperties']
                    break
            if props is None:
                raise ValueError(f'No existe la hoja {sheet_name!r}')
            row_count = props['rowCount']
            col_count = props['columnCount']

            # Función para convertir índice (0-based) a letra Excel
            def col_letter(ix: int) -> str:
                res = ''
                while True:
                    res = chr(65 + (ix % 26)) + res
                    ix = ix // 26 - 1
                    if ix < 0:
                        break
                return res

            last_col = col_letter(col_count - 1)
            start_cell = f'A{header_row}'
            clear_range = f'{sheet_name}!{start_cell}:{last_col}{row_count}'

            # Limpiar únicamente desde header_row hacia abajo
            self.service.spreadsheets().values().clear(
                spreadsheetId=file_id,
                range=clear_range
            ).execute()

            # Preparar los valores: cabecera + filas de datos
            values = [df.columns.tolist()] + df.fillna('').values.tolist()
            body = {'values': values}

            # Escribir empezando en header_row
            update_range = f'{sheet_name}!{start_cell}'
            result = (
                self.service.spreadsheets().values()
                .update(
                    spreadsheetId=file_id,
                    range=update_range,
                    valueInputOption='RAW',
                    body=body
                )
                .execute()
            )
            return result

        except HttpError as e:
            logging.error('Error Sheets API: %s', e)
            raise
        except Exception:
            logging.exception(
                'Error inesperado escribiendo %r en %r',
                sheet_name, file_id
            )
            raise

    def write_spreadsheet(
            self,
            folder_id: str,
            sheet_name: str,
            file_path: str,
            header_row: int = 1
    ) -> str:
        """
        Crea un nuevo Google Spreadsheet dentro de `folder_id` y sobrescribe la hoja
        `sheet_name` con el contenido de `df` (DataFrame de pandas).

        - folder_id: ID de la carpeta contenedora en Drive.
        - sheet_name: nombre de la hoja a crear y poblar.
        - df: DataFrame que contiene los datos a escribir.
        - header_row: (1-based) fila donde colocar los nombres de columna.
          Las filas anteriores permanecen intactas (puede usarse para notas o encabezados manuales).

        Devuelve el ID del spreadsheet generado.
        """
        try:
            # Leer archivo
            df = pd.read_csv(file_path)
            # Inicializar servicio de Sheets
            self._initialize_service()
            # Crear spreadsheet con una hoja inicial nombrada como sheet_name
            metadata = {
                'name': sheet_name,
                'mimeType': 'application/vnd.google-apps.spreadsheet',
                'parents': [folder_id]
            }
            file = self.__drive_service__.files().create(
                body=metadata,
                fields='id'
            ).execute()
            spreadsheet_id = file['id']

            # Renombrar la hoja por defecto (Sheet1) a sheet_name
            batch_update = {
                'requests': [
                    {
                        'updateSheetProperties': {
                            'properties': {'sheetId': 0, 'title': sheet_name},
                            'fields': 'title'
                        }
                    }
                ]
            }
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=batch_update
            ).execute()

            # Preparar datos (cabecera + filas)
            values = [df.columns.tolist()] + df.fillna('').values.tolist()
            start_cell = f'A{header_row}'
            update_range = f'{sheet_name}!{start_cell}'

            # Obtener dimensiones de la hoja para limpiar el rango necesario
            meta = self.service.spreadsheets().get(
                spreadsheetId=spreadsheet_id,
                fields='sheets(properties(title,gridProperties))'
            ).execute()
            props = next(
                (s['properties']['gridProperties'] for s in meta['sheets']
                 if s['properties']['title'] == sheet_name),
                None
            )
            if props is None:
                raise ValueError(f'La hoja {sheet_name!r} no existe en {spreadsheet_id!r}')

            row_count = props['rowCount']
            col_count = props['columnCount']
            last_col = self._col_letter(col_count - 1)
            clear_range = f'{sheet_name}!{start_cell}:{last_col}{row_count}'

            # Limpiar celdas desde header_row hacia abajo
            self.service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=clear_range
            ).execute()

            # Escribir valores en la hoja
            self.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=update_range,
                valueInputOption='RAW',
                body={'values': values}
            ).execute()
            return spreadsheet_id

        except HttpError as e:
            logging.error('Error Google API: %s', e)
            raise
        except Exception as e:
            logging.exception(
                'Error inesperado escribiendo %r en %r: %s',
                sheet_name, folder_id, e
            )
            raise


class SheetsFactory(GoogleFactory):
    def run(self) -> SheetsProduct:
        return SheetsProduct()
