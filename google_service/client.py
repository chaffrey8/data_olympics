from .drive import DriveFactory
from .sheets import SheetsFactory
from .jsons import JsonFactory


class GoogleClient:
    """
    Cliente unificado para servicios de Google: Drive y Sheets.
    Delega llamadas a la clase específica según service_type.
    """

    def __init__(self):
        pass

    def run(self):
        return 'Service'

    def __initialize_drive__(self):
        factory = DriveFactory()
        product = factory.run()
        self.__drive__ = product

    def __initialize_sheet__(self):
        factory = SheetsFactory()
        product = factory.run()
        self.__sheet__ = product

    def __initialize_json__(self):
        factory = JsonFactory()
        product = factory.run()
        self.__json__ = product
    
    def build_drive_service(self):
        factory = DriveFactory()
        product = factory.run()
        return product.service

    def get_files(self, drive_id='root', recursive=False):
        self.__initialize_drive__()
        product = self.__drive__
        return product.get_files(drive_id, recursive)
    
    def create_folder(self, name='Nueva Carpeta', parent_id=None):
        self.__initialize_drive__()
        product = self.__drive__
        return product.create_folder(name, parent_id)

    def download_file(self, file_id, file_path):
        self.__initialize_drive__()
        product = self.__drive__
        return product.download_file(file_id, file_path)

    def upload_file(self, file_path, drive_id, mime_type=None):
        self.__initialize_drive__()
        product = self.__drive__
        return product.upload_file(file_path, drive_id, mime_type)

    def read_spreadsheet(self, spreadsheet_id, sheet_name=None):
        self.__initialize_sheet__()
        product = self.__sheet__
        return product.read_spreadsheet(spreadsheet_id, sheet_name)

    def update_spreadsheet(self, spreadsheet_id, sheet_name, df, header_row=1):
        self.__initialize_sheet__()
        product = self.__sheet__
        return product.update_spreadsheet(spreadsheet_id, sheet_name, df, header_row)
    
    def write_spreadsheet(self, folder_id, sheet_name, file_path, header_row=1):
        self.__initialize_sheet__()
        product = self.__sheet__
        return product.write_spreadsheet(folder_id, sheet_name, file_path, header_row)

    def read_json(self, file_id):
        self.__initialize_json__()
        product = self.__json__
        return product.read_json(file_id)