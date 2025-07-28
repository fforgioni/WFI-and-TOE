# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:35:27 2025

@author: ferfo
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# Cargar el archivo Excel
file_path = "C:/Users/ferfo/OneDrive/Escritorio/metricas media.xlsx"  # Asegúrate de colocar la ruta correcta
xls = pd.ExcelFile(file_path)

# Lista de nombres de modelos
model_names = [
    "CCCma-CanESM2-SMHI", "CNRM-CERFACS-CNRM-CM5-SMHI", "CSIRO-QCCCE-CSIRO-Mk3-6-0-SMHI", "Ensamble",
    "ICHEC-EC-EARTH-SMHI", "IPSL-IPSL-CM5A-MR-SMHI", "MIROC-MIROC5-SMHI", "MOHC-HadGEM2-ES-SMHI",
    "MPI-M-MPI-ESM-LR-SMHI", "NCC-NorESM1-M-SMHI", "NOAA-GFDL-GFDL-ESM2M-SMHI"
]

# Extraer y organizar los datos de Correlation, RMSE y Log Std Ratio
metric_data = {"Correlation": {}, "RMSE": {}, "Log Std Ratio": {}}

for sheet in xls.sheet_names:
    df = xls.parse(sheet)
    
    # Verificar si las columnas necesarias existen
    for metric in metric_data.keys():
        for col in df.columns:
            if metric.lower() in col.lower():  # Buscar la columna ignorando mayúsculas
                df.rename(columns={col: metric}, inplace=True)
        if metric in df.columns and not df[metric].dropna().empty:
            metric_data[metric][sheet.replace('_Mean', '').replace('_P90', '')] = df[metric].values  # Remover '_Mean' y 'P90'

# Convertir a DataFrame para visualización
correlation_df = pd.DataFrame(metric_data["Correlation"])
rmse_df = pd.DataFrame(metric_data["RMSE"])
log_std_ratio_df = pd.DataFrame(metric_data["Log Std Ratio"])

# Asignar nombres de modelos a las filas
for df in [correlation_df, rmse_df, log_std_ratio_df]:
    if not df.empty:
        df.index = model_names[:len(df)]

# Crear el gráfico con 3 subplots (Correlation, RMSE y Log Std Ratio)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 20))

# Configuración de cada subplot con diferentes paletas de colores #BrBG, twilight_r, pink
sns.heatmap(correlation_df.T, annot=True, cmap="Blues", linewidths=0.5, vmin=0, vmax=1, 
            cbar=False, annot_kws={"size": 20, "color": "white", "weight": "bold"}, ax=axes[0])
axes[0].set_title("", fontsize=20)
axes[0].set_xlabel("")
axes[0].set_ylabel("(a) Correlation", fontsize=30, labelpad = 20)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=12)

sns.heatmap(rmse_df.T, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, cbar=False, 
            annot_kws={"size": 20, "color": "white", "weight": "bold"}, ax=axes[1])
axes[1].set_title("", fontsize=18)
axes[1].set_xlabel("")
axes[1].set_ylabel("(b) RMSE", fontsize=30, labelpad = 20)
axes[1].set_xticklabels([])

sns.heatmap(log_std_ratio_df.T, annot=True, fmt=".2f", cmap="Greens", linewidths=0.5, cbar=False, 
            annot_kws={"size": 20, "color": "white", "weight": "bold"}, ax=axes[2])
axes[2].set_title("", fontsize=18)
axes[2].set_xlabel("")
axes[2].set_ylabel("(c) Log Std Ratio", fontsize=30, labelpad = 20)
axes[2].set_xticklabels([])

# Ajustar etiquetas de ejes
for ax in axes:
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)

# Aplicar cambio de color de texto a blanco y añadir contorno
for ax in axes:
    for text in ax.texts:
        text.set_color("Black")
     

# Guardar la figura
plt.tight_layout()
plt.savefig("C:/Users/ferfo/OneDrive/Escritorio/Heatmap media.png", dpi=300, bbox_inches='tight')
plt.show()
