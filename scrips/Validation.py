# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 10:41:53 2025

@author: ferfo
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
from glob import glob

# Ruta base de las carpetas
base_folder = "C:/Users/ferfo/OneDrive/Escritorio/"

# Definir las variables, sus respectivas carpetas y archivos CEMS
variables = {
    "FFMC": ("ffmcode", "FFMC", "FFMC", "C:/Users/ferfo/OneDrive/Escritorio/FFMC/FFMC-cems-1.nc"),
    "DMC": ("dufmcode", "DMC", "DMC", "C:/Users/ferfo/OneDrive/Escritorio/DMC/DMC-cems-1.nc"),
    "DC": ("drtcode", "DC", "DC", "C:/Users/ferfo/OneDrive/Escritorio/DC/DC-cems-1.nc"),
    "ISI": ("infsinx", "ISI", "ISI", "C:/Users/ferfo/OneDrive/Escritorio/ISI/ISI-cems-1.nc"),
    "BUI": ("fbupinx", "BUI", "BUI", "C:/Users/ferfo/OneDrive/Escritorio/BUI/BUI-cems-1.nc"),
    "FWI": ("fwinx", "FWI", "FWI", "C:/Users/ferfo/OneDrive/Escritorio/FWI/FWI-cems-1.nc"),
}

# Diccionarios para almacenar resultados por mes
results_mean = {var: {m: [] for m in range(1, 13)} for var in variables}
results_p90 = {var: {m: [] for m in range(1, 13)} for var in variables}

# Procesar cada índice en su carpeta correspondiente
for var, (cems_var, model_var, folder, cems_file) in variables.items():
    cems_path = os.path.join(base_folder, cems_file)
    
    if not os.path.exists(cems_path):
        print(f"⚠️ Archivo CEMS no encontrado: {cems_path}")
        continue
    
    # Cargar el dataset CEMS específico
    ds_cems = xr.open_dataset(cems_path)
    model_folder = os.path.join(base_folder, folder)
    model_files = glob(os.path.join(model_folder, "*.nc"))
    
    for model_file in model_files:
        try:
            print(f"Procesando {var}: {model_file}")

            # Cargar el modelo
            ds_model = xr.open_dataset(model_file)

            if cems_var in ds_cems and model_var in ds_model:
                # Seleccionar las variables correspondientes
                cems_data = ds_cems[cems_var]
                model_data = ds_model[model_var]
                
                # Asegurar que las dimensiones coincidan con CEMS
                model_data = model_data.sel(time=cems_data.time, latitude=cems_data.latitude, longitude=cems_data.longitude)

                for month in range(1, 13):
                    # Filtrar por mes
                    cems_monthly = cems_data.sel(time=cems_data.time.dt.month == month)
                    model_monthly = model_data.sel(time=model_data.time.dt.month == month)
                    
                    if len(cems_monthly.time) == 0 or len(model_monthly.time) == 0:
                        continue
                    
                    cems_mean = cems_monthly.mean(dim="time")
                    model_mean = model_monthly.mean(dim="time")
                    
                    cems_p90 = cems_monthly.reduce(np.percentile, q=90, dim="time")
                    model_p90 = model_monthly.reduce(np.percentile, q=90, dim="time")

                    # Asegurar alineación
                    cems_mean, model_mean = xr.align(cems_mean, model_mean, join="inner")
                    cems_p90, model_p90 = xr.align(cems_p90, model_p90, join="inner")

                    # Crear máscara para eliminar NaNs
                    nan_mask_mean = np.isnan(cems_mean.values) | np.isnan(model_mean.values)
                    cems_clean_mean = cems_mean.values[~nan_mask_mean]
                    model_clean_mean = model_mean.values[~nan_mask_mean]

                    nan_mask_p90 = np.isnan(cems_p90.values) | np.isnan(model_p90.values)
                    cems_clean_p90 = cems_p90.values[~nan_mask_p90]
                    model_clean_p90 = model_p90.values[~nan_mask_p90]

                    # Calcular métricas con Bias normalizado
                    def calc_metrics(clean_cems, clean_model):
                        if clean_cems.size > 1 and clean_model.size > 1:
                            cc = np.corrcoef(clean_cems.flatten(), clean_model.flatten())[0, 1]
                            rmse = np.sqrt(np.mean((clean_cems - clean_model) ** 2))
                            bias = np.mean(clean_cems - clean_model)
                            
                            # Bias normalizado (%)
                            if np.sum(clean_cems) != 0:
                                bias_pct = (np.sum(clean_cems - clean_model) / np.sum(clean_cems)) * 100
                            else:
                                bias_pct = np.nan
                        else:
                            cc, rmse, bias, bias_pct = np.nan, np.nan, np.nan, np.nan
                        return cc, rmse, bias, bias_pct
                    
                    # Guardar resultados por mes
                    results_mean[var][month].append([os.path.basename(model_file), *calc_metrics(cems_clean_mean, model_clean_mean)])
                    results_p90[var][month].append([os.path.basename(model_file), *calc_metrics(cems_clean_p90, model_clean_p90)])

            ds_model.close()
        except Exception as e:
            print(f"❌ Error procesando {model_file}: {e}")

# Crear archivos Excel con hojas separadas por mes
output_mean = os.path.join(base_folder, "model_metrics_mean_monthly_normalized.xlsx")
output_p90 = os.path.join(base_folder, "model_metrics_p90_monthly_normalized.xlsx")

with pd.ExcelWriter(output_mean) as writer:
    for var, months_data in results_mean.items():
        for month, data in months_data.items():
            df = pd.DataFrame(data, columns=["Model", "Correlation", "RMSE", "Bias", "Bias (%)"])
            df.to_excel(writer, sheet_name=f"{var}_M{month:02d}", index=False)

with pd.ExcelWriter(output_p90) as writer:
    for var, months_data in results_p90.items():
        for month, data in months_data.items():
            df = pd.DataFrame(data, columns=["Model", "Correlation", "RMSE", "Bias", "Bias (%)"])
            df.to_excel(writer, sheet_name=f"{var}_M{month:02d}", index=False)

print(f"✅ Resultados guardados en:\n - {output_mean}\n - {output_p90}")
