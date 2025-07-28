# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:55:59 2025

@author: ferfo
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Ruta base donde están los archivos
base_path = "C:/Users/ferfo/OneDrive/Escritorio/a"
indices = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

# Diccionario para almacenar resultados
results = {"Model": [], "Index": [], "Correlation": [], "RMSE": [], "Bias": [], "Std Ratio": []}

for index in indices:
    index_path = os.path.join(base_path, index)
    if not os.path.exists(index_path):
        continue  # Saltar si la carpeta no existe
    
    files = [f for f in os.listdir(index_path) if f.endswith(".nc")]
    
    # Identificar archivo de CEMS
    cems_file = next((f for f in files if "cems" in f.lower()), None)
    if not cems_file:
        continue  # Saltar si no hay archivo CEMS
    
    cems_data = xr.open_dataset(os.path.join(index_path, cems_file))
    
    # Verificar nombres de dimensiones
    dims = list(cems_data.dims.keys())
    lat_dim = "latitude" if "latitude" in dims else "lat"
    lon_dim = "longitude" if "longitude" in dims else "lon"
    
    cems_mean = cems_data[list(cems_data.data_vars.keys())[0]].mean(dim=[lat_dim, lon_dim])
    
    for file in files:
        if "cems" in file.lower() or "ensamble" in file.lower():
            continue  # Saltar CEMS y Ensamble
        
        model_name = file.replace(f"{index}-", "").replace(".nc", "")
        model_data = xr.open_dataset(os.path.join(index_path, file))
        
        model_mean = model_data[list(model_data.data_vars.keys())[0]].mean(dim=[lat_dim, lon_dim])
        
        # Calcular métricas
        correlation, _ = spearmanr(model_mean.values, cems_mean.values)
        rmse = np.sqrt(np.mean((model_mean.values - cems_mean.values) ** 2))
        bias = np.mean(model_mean.values - cems_mean.values)
        std_ratio = np.std(model_mean.values) / np.std(cems_mean.values)
        
        # Guardar resultados
        results["Model"].append(model_name)
        results["Index"].append(index)
        results["Correlation"].append(correlation)
        results["RMSE"].append(rmse)
        results["Bias"].append(bias)
        results["Std Ratio"].append(std_ratio)

# Convertir resultados en DataFrame
results_df = pd.DataFrame(results)

# Determinar los modelos en los terciles superior e inferior
upper_tercile = results_df.groupby("Index")["Correlation"].transform(lambda x: x >= x.quantile(0.66))
lower_tercile = results_df.groupby("Index")["Correlation"].transform(lambda x: x <= x.quantile(0.33))

results_df["Upper"] = upper_tercile.astype(int)  # Convertir a 1 y 0
results_df["Lower"] = lower_tercile.astype(int)  # Convertir a 1 y 0

# Crear conteos por modelo e índice
upper_count = results_df[results_df["Upper"] == 1].groupby(["Model", "Index"]).size().unstack(fill_value=0)
lower_count = results_df[results_df["Lower"] == 1].groupby(["Model", "Index"]).size().unstack(fill_value=0)

# Visualización estilo del paper
fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [4, 1]})

sns.heatmap(upper_count, cmap=sns.light_palette('#6495ED', as_cmap=True), ax=axes[0, 0], linewidths=0.5, linecolor='black', annot=False, cbar=False)
axes[0, 0].set_title("(a) Upper-tercile count", fontsize = 22)
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("")
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), fontsize=14)  # Ajusta el tamaño de las etiquetas X
axes[0, 0].set_yticklabels(axes[0, 0].get_yticklabels(), fontsize=14) 


axes[0, 1].barh(upper_count.index[::-1], upper_count.sum(axis=1).values[::-1], color='#6495ED')
axes[0, 1].set_yticks(np.arange(0, len(upper_count.index), 1))
axes[0, 1].set_yticklabels([])
#axes[0, 1].set_title("Total upper-tercile count")
axes[0, 1].set_xticks(np.arange(0, max(upper_count.sum(axis=1).values)+1, 1))


sns.heatmap(lower_count, cmap=sns.light_palette('#CD5C5C', as_cmap=True), ax=axes[1, 0], linewidths=0.5, linecolor='black', annot=False, cbar=False)
axes[1, 0].set_title("(b) Lower-tercile count", fontsize = 22)
axes[1, 0].set_xlabel("")
axes[1, 0].set_ylabel("")
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), fontsize=14)
axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels(), fontsize=14)

axes[1, 1].barh(lower_count.index[::-1], lower_count.sum(axis=1).values[::-1], color='#CD5C5C')
axes[1, 1].set_yticks(np.arange(0, len(lower_count.index), 1))
axes[1, 1].set_yticklabels([])
#axes[1, 1].set_title("Total lower-tercile count")
axes[1, 1].set_xticks(np.arange(0, max(lower_count.sum(axis=1).values)+1, 1))

plt.tight_layout()
plt.savefig("C:/Users/ferfo/OneDrive/Escritorio/tercile_count_plot.png", dpi=300, bbox_inches="tight")  # Guarda en PNG con alta calidad
plt.show()

