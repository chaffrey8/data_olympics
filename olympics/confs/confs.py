from __init__ import out_dir

# Rutas necesarias del drive
folders_id = {

}

# Diccionario con la configuración necesaria para 
# API de Google
google_confs = {
    'scopes': [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
    ],
    'client_filename': 'credentials.json',
    'token_filename': 'token.json',
}

# Bóveda archivos
safehouse = out_dir / 'safehouse'