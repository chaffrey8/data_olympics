from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import mimetypes
from pathlib import Path

from .base import GoogleFactory, GoogleProduct


class DriveProduct(GoogleProduct):
    def __init__(self) -> None:
        super().__init__('drive')
        self.type = 'drive'
        self._initialize_service()

    def run(self) -> any:
        return 'Service'

    def _initialize_service(self):
        self.service = build('drive', 'v3', credentials=self.creds)

    def get_files(self, drive_id: str = 'root', recursive: bool = False) -> list:
        """
        Lista los archivos/directorios que cuelgan de `drive_id`.
        Si recursive=True, baja dentro de cada carpeta y sigue listando.
        """
        all_items = []
        page_token = None
        try:
            self._initialize_service()
            while True:
                response = (
                    self.service.files()
                    .list(
                        q=f'"{drive_id}" in parents and trashed=false',
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                        pageSize=100,
                        fields='nextPageToken, files(id, name, mimeType, '
                        'size, quotaBytesUsed)',
                        pageToken=page_token,
                    )
                    .execute()
                )
                items = response.get('files', [])
                for item in items:
                    all_items.append(item)
                    # Si es carpeta y queremos recursividad,
                    # llamamos de nuevo a get_files sobre su ID
                    mt = 'application/vnd.google-apps.folder'
                    if recursive and item.get('mimeType') == mt:
                        sub_items = self.get_files(drive_id=item['id'], recursive=True)
                        all_items.extend(sub_items)
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
        except HttpError as error:
            print(f'An error occurred: {error}')
        return all_items

    def download_file(self, file_id: str, file_path: str) -> any:
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_path, mode="wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%")
        return fh

    def upload_file(
        self,
        file_path: str,
        drive_id: str,
        mime_type: str = None
    ) -> dict:
        """
        Sube un archivo local a la carpeta de Drive indicada por drive_id.

        Parámetros:
        - file_path: ruta al fichero en tu disco.
        - drive_id: ID de la carpeta de Drive donde se quiere subir.
        - mime_type: tipo MIME del archivo (se intenta adivinar si no se proporciona).

        Devuelve:
        - metadata del fichero subido (id, name, mimeType, size, …).
        """
        try:
            # Asegurarnos de tener el servicio inicializado
            self._initialize_service()
            # Nombre y mimeType
            file_name = Path(file_path).stem
            if not mime_type:
                mime_type = (
                    mimetypes.guess_type(file_path)[0]
                    or
                    'application/octet-stream'
                )
            # Preparar metadatos y contenido
            file_metadata = {
                'name': file_name,
                'mimeType': mime_type,
                'parents': [drive_id]
            }
            media = MediaFileUpload(
                file_path,
                mimetype=mime_type,
                resumable=True
            )
            # Llamada a la API para crear el fichero en Drive
            request = self.service.files().create(
                body=file_metadata,
                media_body=media,
                supportsAllDrives=True,
                fields='id, name, mimeType, size'
            )
            # Subida resumable (permite seguimiento de progreso si se desea)
            uploaded_file = None
            while uploaded_file is None:
                status, uploaded_file = request.next_chunk()
                if status:
                    print(f'Uploaded {int(status.progress() * 100)}%.')
            print(f'Subida completada: {uploaded_file.get("id")}')
            return uploaded_file
        except HttpError as error:
            print(f'Error al subir el fichero: {error}')
            raise

    def create_folder(
            self,
            name: str = 'Nueva carpeta',
            parent_id: str = None
    ) -> dict:
        """
        Crea una carpeta llamada 'name' en la raíz de Drive
        y devuelve el recurso (incluyendo su ID).
        """
        try:
            metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id is not None:
                metadata['parents'] = [parent_id]
            folder = self.service.files().create(
                body=metadata,
                fields='id'
            ).execute()
            print(f'Carpeta {name} creada con ID: {folder.get("id")}')
            return folder.get('id')
        except HttpError as error:
            print(f'Error al crear la carpeta: {error}')
            raise

    def select_folder(self) -> str:
        """
        Permite al usuario navegar por las carpetas de su Drive
        y devuelve el ID de la carpeta seleccionada.
        """
        parent_id = 'root'
        path = []

        while True:
            # 1) Listar carpetas bajo el folder actual
            resp = self.service.files().list(
                q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                spaces='drive',
                fields='files(id, name)',
                pageSize=100
            ).execute()
            folders = resp.get('files', [])

            # 2) Mostrar navegación actual
            current_path = ' / '.join(path) or 'Mi unidad'
            print(f"\nCarpetas en: {current_path}")
            for i, f in enumerate(folders, start=1):
                print(f"{i}. {f['name']}")

            print("0. Seleccionar carpeta actual")
            if path:
                print("U. Subir un nivel")
            print("Q. Cancelar")

            # 3) Leer elección
            choice = input("Elige (número, 0, U o Q): ").strip().lower()
            if choice == 'q':
                print("Selección cancelada.")
                return None

            if choice == '0':
                print(f"Carpeta seleccionada: {current_path} (ID={parent_id})")
                return parent_id

            if choice == 'u' and path:
                # Subir un nivel: obtener padre del folder actual
                meta = self.service.files().get(
                    fileId=parent_id,
                    fields='parents'
                ).execute()
                parents = meta.get('parents', [])
                parent_id = parents[0] if parents else 'root'
                path.pop()
                continue

            # Si es número, bajar al subfolder elegido
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(folders):
                    sel = folders[idx]
                    parent_id = sel['id']
                    path.append(sel['name'])
                else:
                    print("Número fuera de rango.")
            except ValueError:
                print("Entrada no válida. Usa un número, U, 0 o Q.")

class DriveFactory(GoogleFactory):
    def run(self) -> DriveProduct:
        return DriveProduct()
