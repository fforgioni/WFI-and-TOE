# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 23:39:38 2025

@author: ferfo
"""

import numpy as np
import xarray as xr
import math

def DMC_numpy(TEMP, RH, RAIN, DMCPrev, LAT):
    RH = np.minimum(100.0, RH)
    re = np.where(RAIN > 5.0, 1.0 * RAIN - 1.5, 0)

    mo = 20.0 + np.exp(5.6348 - DMCPrev / 43.43)

    b = np.where(DMCPrev <= 33.0,
                 100.0 / (0.5 + 0.3 * DMCPrev),
                 np.where(DMCPrev <= 65.0,
                          14.0 - 1.3 * np.log(np.maximum(DMCPrev, 0.0001)),
                          6.2 * np.log(np.maximum(DMCPrev, 0.0001)) - 17.2))

    mr = mo + 1000.0 * re / (48.77 + b * re)
    pr = np.where(mr > 20.0, 244.72 - 43.43 * np.log(np.maximum(mr - 20.0, 0.0001)), 0.0)
    DMCPrev = np.where(RAIN > 5.0, np.maximum(pr, 0.0), DMCPrev)
    k = np.where(TEMP > -1.1, 1.894 * (TEMP + 1.1) * (100.0 - RH) * 0.00003, 0.0)

    return DMCPrev + 100.0 * k

# 📂 Cargar datos
ds_temp = xr.open_dataset("E:/tas-max/tasmax-MOHC-HadGEM2-ES-SMHI-RCA4.nc")
ds_hum = xr.open_dataset("E:/hurs-min/hurs-MOHC-HadGEM2-ES-SMHI-RCA4.nc")
ds_precip = xr.open_dataset("E:/pp/pr-MOHC-HadGEM2-ES-SMHI-RCA4-corregido.nc")

T_max = ds_temp["tas"] - 273.15
H_min = ds_hum["hurs"]
P = ds_precip["pr"] * 86400

# 🧠 Convertir a NumPy arrays
temp_vals = T_max.values
hum_vals = H_min.values
rain_vals = P.values
lat_vals = ds_temp["lat"].values
time_vals = ds_temp.time.values

# 🧱 Inicializar
ntime, nlat, nlon = temp_vals.shape
DMC_prev = np.full((nlat, nlon), 6.0, dtype=np.float32)
DMC_all = np.empty((ntime, nlat, nlon), dtype=np.float32)

# 🔁 Loop temporal optimizado
for t in range(ntime):
    if t % 100 == 0:
        print(f"Procesando día {t}/{ntime}")

    T = temp_vals[t]
    RH = hum_vals[t]
    P_day = rain_vals[t]

    DMC_today = DMC_numpy(T, RH, P_day, DMC_prev, lat_vals[:, None])
    DMC_all[t] = DMC_today
    DMC_prev = DMC_today  # estado para el siguiente día

# 📦 Crear Dataset xarray
ds_dmc = xr.Dataset(
    {"DMC": (["time", "rlat", "rlon"], DMC_all)},
    coords={
        "time": ds_temp["time"],
        "rlat": ds_temp["rlat"],
        "rlon": ds_temp["rlon"],
        "lat": ds_temp["lat"],
        "lon": ds_temp["lon"]
    }
)

# 💾 Guardar
output_file = "C:/Users/ferfo/OneDrive/Escritorio/DMC-MOHC-HadGEM2-ES-SMHI-RCA4.nc"
ds_dmc.to_netcdf(output_file)
print(f"✅ ¡DMC calculado y guardado en {output_file}!")

# 📁 Cerrar
ds_temp.close()
ds_hum.close()
ds_precip.close()
