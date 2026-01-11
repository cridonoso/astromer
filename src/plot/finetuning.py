import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import re


def _parse_label_from_path(path):
    """Extrae etiqueta legible del path (ej: ATLAS 20)."""
    match = re.search(r'(alcock|atlas)_(\d+)', path.lower())
    if match:
        name = match.group(1)
        spc = match.group(2)
        return name, spc
    
    # Intento fallback para pretraining si dice 'pretraining'
    if 'pretraining' in path.lower():
        return "Pretraining"
    return "Unknown"

def get_aggregated_statistics(df_raw, transform_func=None):
    """
    Calcula media y std para la métrica y el tiempo.
    Robustecida para desempaquetar arrays de numpy dentro de las celdas.
    """
    if df_raw.empty:
        print("El DataFrame de entrada está vacío.")
        return pd.DataFrame()

    df_proc = df_raw.copy()

    # --- PASO 1: DESEMPAQUETADO PROFUNDO ---
    # Esta función extrae el escalar si el valor es un array de numpy o un tipo genérico de numpy
    def unwrap_scalar(x):
        if isinstance(x, (np.ndarray, np.generic)):
            return x.item()
        return x

    # Aplicamos el desempaquetado celda por celda
    df_proc['value'] = df_proc['value'].apply(unwrap_scalar)

    # --- PASO 2: CONVERSIÓN A FLOAT ---
    # Ahora que son escalares, forzamos la conversión a float nativo
    df_proc['value'] = pd.to_numeric(df_proc['value'], errors='coerce')

    # --- PASO 3: TRANSFORMACIÓN (SQRT/SQUARE) ---
    if transform_func is not None:
        try:
            # Ahora transform_func recibe una serie de floats limpios
            df_proc['value'] = transform_func(df_proc['value'])
        except Exception as e:
            print(f"Error aplicando transform_func: {e}")
            return pd.DataFrame()

    # --- PASO 4: AGREGACIÓN ---
    time_col = 'wall_time' if 'wall_time' in df_proc.columns else 'walltime'
    
    # Aseguramos limpieza en el tiempo también
    if time_col in df_proc.columns:
        df_proc[time_col] = df_proc[time_col].apply(unwrap_scalar)
        df_proc[time_col] = pd.to_numeric(df_proc[time_col], errors='coerce')
    
    aggregations = {
        'value': ['mean', 'std'],
        time_col: ['mean', 'std']
    }

    # Eliminamos filas donde la métrica sea NaN antes de agrupar
    agg_df = df_proc.dropna(subset=['value']).groupby(['data', 'spc']).agg(aggregations).reset_index()
    
    agg_df.columns = [
        'dataset', 
        'spc', 
        'metric_mean', 
        'metric_std', 
        'time_mean_sec', 
        'time_std_sec'
    ]
    
    agg_df = agg_df.sort_values(by=['dataset', 'spc'])
    
    return agg_df

import pandas as pd
import numpy as np
import toml
import os
import re

def extract_final_metrics_to_df(finetune_paths, loader_func):
    """
    Extrae el RMSE desde 'test_metrics.toml' y la duración total del entrenamiento
    desde los logs de TensorBoard.
    """
    records = []
    
    for path in finetune_paths:
        # --- 1. Extraer Tiempo (desde TensorBoard) ---
        duration = 0.0
        try:
            # Usamos smart_load_logs solo para sacar el tiempo
            logs = loader_func(path)
            
            if logs is not None and logs[1] is not None:
                _, val_df = logs
                if 'wall_time' in val_df.columns:
                    # Tiempo total = Último registro - Primer registro
                    start_time = val_df['wall_time'].min()
                    end_time = val_df['wall_time'].max()
                    duration = end_time - start_time
        except Exception as e:
            print(f"Advertencia: No se pudo calcular tiempo para {path} ({e})")
            duration = 0.0

        # --- 2. Extraer RMSE (desde test_metrics.toml) ---
        toml_path = os.path.join(path, 'test_metrics.toml')
        
        # Si no existe en la ruta directa, intentamos buscar si la ruta fue corregida
        # (Esto asume que el toml vive junto al config.toml)
        if not os.path.exists(toml_path):
            print(f"Saltando {path}: No se encontró 'test_metrics.toml'")
            continue

        try:
            with open(toml_path, 'r') as f:
                data = toml.load(f)
            
            # Buscamos la llave 'rmse' (o 'test_rmse' como fallback común)
            rmse_val = data.get('rmse')
            if rmse_val is None:
                rmse_val = data.get('test_rmse')
                
            if rmse_val is None:
                print(f"Saltando {path}: El TOML no contiene la llave 'rmse'")
                continue
                
            final_val = float(rmse_val)
            
        except Exception as e:
            print(f"Error leyendo TOML en {path}: {e}")
            continue

        # --- 3. Parsear Metadata (Dataset y SPC) ---
        dataset_name = 'Unknown'
        spc_val = 'Unknown'
        
        # Regex para Atlas/Alcock
        match = re.search(r'(alcock|atlas)', path.lower())
        if match: 
            dataset_name = match.group(1)
        
        # Regex para SPC (ej: _20, _100)
        match_spc = re.search(r'_(\d+)', path)
        if match_spc: 
            spc_val = int(match_spc.group(1))
        
        records.append({
            'data': dataset_name,
            'spc': spc_val,
            'value': final_val,       # Este es el RMSE del TOML
            'wall_time': duration     # Tiempo en segundos desde Tensorboard
        })
            
    return pd.DataFrame(records)

def get_pretrain_baseline(pretrain_path, loader_func, metric_col='loss'):
    """Obtiene el valor escalar del pretraining para usar de referencia."""
    logs = loader_func(pretrain_path)
    if logs and logs[1] is not None:
        val_df = logs[1]
        if metric_col in val_df.columns:
            return val_df[metric_col].min()
    return None