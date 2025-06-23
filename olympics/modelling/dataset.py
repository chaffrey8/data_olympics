from datetime import datetime
import pandas as pd
from pathlib import Path
from workalendar.america import Mexico

from ..google_service.client import GoogleClient

class WFMDataset:
    '''
    Clase para trabajar con bases de datos de Work Force México:
    - data_path:            ruta en donde se localiza la base de datos con la
                            que se va a trabajar.
    - outlier_treatment:    método para tratar los datos atípicos, puede ser
                            "mad", "z-score", "iqr" o None. Si se selecciona
                            None (valor por defecto) no se implementa ningún
                            tratamiento.
    - imputation_method:    método a utilizar para imputar datos faltantes,
                            puede ser "knn", "bayesian_ridge", "random_forest"
                            o "interpolation" (valor por defecto).
    - group_by_service:     bandera que indica si se debe utilizar el servicio
                            como una variable agrupativa. En caso de ser True
                            (valor por defecto) se da tratamiento independiente
                            a los datos para cada servicio.
    '''
    def __init__(
            self,
            spreadsheet_id: str,
            sheet_name: str,
            outlier_treatment: str = None,
            imputation_method: str = 'interpolation',
            group_by_service: bool = True
    ) -> None:
        self.data = self.read_data(spreadsheet_id, sheet_name)
        self.__outlier_treatment__ = outlier_treatment
        self.__imputation_method__ = imputation_method
        self.__group_by_service__ = group_by_service
        self.__calendar__ = Mexico()

    def read_data(self, spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
        """
        Lee un archivo CSV y devuelve un DataFrame limpio y tipado.

        Se omite la primera fila vacía del archivo y se eliminan filas vacías.
        Se validan columnas obligatorias ('FECHA', 'SERVICIO') y opcionales
        ('FC 30', 'AJ SEM', 'RECIBIDAS'), garantizando tipos correctos.

        Parameters
        ----------
        data_path : Path
            Ruta al archivo CSV que se desea leer.

        Returns
        -------
        pd.DataFrame
            DataFrame con las columnas ['FECHA', 'SERVICIO', 'FC 30',
            'AJ SEM', 'RECIBIDAS'], con tipos:
            - FECHA: datetime64[ns]
            - SERVICIO: categoría
            - FC 30, AJ SEM, RECIBIDAS: enteros nulos (Int64)

        Raises
        ------
        ValueError
            Si faltan las columnas obligatorias 'FECHA' o 'SERVICIO'.

        Notes
        -----
        - Las fechas mal formateadas se convierten en NaT.
        - Las columnas opcionales ausentes se crean con valores nulos.
        """
        # Leer la base
        cliente = GoogleClient()
        df = cliente.read_spreadsheet(
            spreadsheet_id=spreadsheet_id,
            sheet_name=sheet_name,

        )
        # Eliminar renglones vacíos
        df = df.dropna(how='all')
        # Garantizar que existan las columnas requeridas
        required_columns = ['FECHA', 'SERVICIO']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        # Garantizar que existan las columnas opcionales en el dataframe
        optional_columns = ['FC 30', 'AJ SEM', 'RECIBIDAS']
        for col in optional_columns:
            if col not in df.columns:
                df[col] = pd.NA
        # Formatear tipo de variables
        df['FECHA'] = pd.to_datetime(
            df['FECHA'],
            format='%d/%m/%Y',
            errors='coerce'
        )
        df['SERVICIO'] = df['SERVICIO'].astype('category')
        for col in optional_columns:
            df[col] = (
                pd.to_numeric(df[col], errors='coerce')
                .round()
                .astype('Int64')
            )
        # Conservar solo las columnas necesarias en el orden adecuado
        final_columns = ['FECHA', 'SERVICIO'] + optional_columns
        df = df[final_columns]
        # Ordenar observaciones
        df = df.sort_values('FECHA').reset_index(drop=True)
        return df

    def train_test_split(
            self,
            cutpoint_date: datetime,
            servicio: str = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.data.copy()
        if (cutpoint_date <= df['FECHA'].min()) or (cutpoint_date >= df['FECHA'].max()):
            raise ValueError("El punto de corte debe ser una fecha dentro del rango de las observaciones.")
        df_train = df[df['FECHA'] <= cutpoint_date]
        df_test = df[df['FECHA'] > cutpoint_date]
        if servicio is not None and servicio in df['SERVICIO'].values:
            df_train = df_train[df_train['SERVICIO']==servicio]
            df_train = df_train.drop(columns=['SERVICIO'])
            df_train = df_train.reset_index(drop=True)
            df_test = df_test[df_test['SERVICIO']==servicio]
            df_test = df_test.drop(columns=['SERVICIO'])
            df_test = df_test.reset_index(drop=True)
        return df_train, df_test