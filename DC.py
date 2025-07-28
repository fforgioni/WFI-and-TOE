# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 01:24:48 2025

@author: ferfo
"""

import numpy as np
import xarray as xr
import pandas as pd

def DC_numpy(TEMP, RAIN, DCPrev, LAT, MONTH):
    """Versión vectorizada del cálculo del Drought Code (DC)."""
    
    DC_new = DCPrev.copy()
    mask_rain = RAIN > 2.8

    if np.any(mask_rain):
        rd = 0.83 * RAIN[mask_rain] - 1.27
        Qo = 800.0 * np.exp(-DCPrev[mask_rain] / 400.0)
        Qr = Qo + 3.937 * rd
        Dr = np.where(Qr > 0, 400.0 * np.log(800.0 / np.maximum(Qr, 0.0001)), 0.0)
        DC_new[mask_rain] = np.maximum(Dr, 0.0)

    # Secado
    Lf = DryingFactor_numpy(LAT, MONTH)
    V = np.where(TEMP > -2.8, 0.36 * (TEMP + 2.8) + Lf, Lf)
    V = np.maximum(V, 0.0)

    return DC_new + 0.5 * V

def DryingFactor_numpy(Latitude, Month):
    """Factor de secado vectorizado."""
    LfN = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])
    LfS = np.array([6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8])

    idx = Month.astype(int) - 1  # asegurar índice 0-11
    return np.where(Latitude > 0, LfN[idx], LfS[idx])

# 📂 Cargar datos
ds_temp = xr.open_dataset("E:/tas-max/tasmax-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4.nc")
ds_precip = xr.open_dataset("E:/pp/pr-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4-corregido.nc")

T_max = ds_temp["tas"] - 273.15
P = ds_precip["pr"] * 86400

# 🔧 Convertir a NumPy arrays
temp_vals = T_max.values
rain_vals = P.values
lat_vals = ds_temp["lat"].values
time_vals = ds_temp["time"].values
ntime, nlat, nlon = temp_vals.shape

# 🧱 Inicializar
DC_prev = np.full((nlat, nlon), 15.0, dtype=np.float32)
DC_all = np.empty((ntime, nlat, nlon), dtype=np.float32)

# 🔁 Loop temporal optimizado
for t in range(ntime):
    if t % 100 == 0:
        print(f"Procesando DC {t}/{ntime}")

    T = temp_vals[t]
    R = rain_vals[t]
    #month = pd.to_datetime(str(time_vals[t])).month
    month = ds_temp["time"].dt.month.values[t]

    # Crear grillas para latitud y mes
    if lat_vals.ndim == 1:
        # Caso típico: latitud como vector 1D
        lat_grid = np.broadcast_to(lat_vals[:, np.newaxis], (nlat, nlon))
    else:
        # Caso: latitud ya está en forma 2D (por ejemplo, (rlat, rlon))
        lat_grid = lat_vals

    month_grid = np.full_like(lat_grid, month)

    DC_today = DC_numpy(T, R, DC_prev, lat_grid, month_grid)
    DC_all[t] = DC_today
    DC_prev = DC_today

# 📦 Crear Dataset
ds_dc = xr.Dataset(
    {"DC": (["time", "rlat", "rlon"], DC_all)},
    coords={
        "time": ds_temp["time"],
        "rlat": ds_temp["rlat"],
        "rlon": ds_temp["rlon"],
        "lat": ds_temp["lat"],
        "lon": ds_temp["lon"]
    }
)

# 💾 Guardar con compresión
output_file = "C:/Users/ferfo/OneDrive/Escritorio/DC-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4.nc"
ds_dc.to_netcdf(output_file, encoding={"DC": {"zlib": True, "complevel": 4}})
print(f"✅ ¡DC calculado y guardado en {output_file}!")

# 📁 Cerrar archivos
ds_temp.close()
ds_precip.close()
