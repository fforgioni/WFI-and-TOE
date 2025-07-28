# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 00:17:02 2025

@author: ferfo
"""

import numpy as np
import xarray as xr

def ISI_numpy(WIND, FFMC):
    """Versión vectorizada del ISI."""
    fWIND = np.exp(0.05039 * WIND)
    m = 147.2 * (101.0 - FFMC) / (59.5 + FFMC)
    fF = 91.9 * np.exp(-0.1386 * m) * (1.0 + np.power(m, 5.31) / 49300000.0)
    return 0.208 * fWIND * fF

# 📂 Cargar datasets
wind_ds = xr.open_dataset("E:viento/sfcwind-ICHEC-EC-EARTH-SMHI-RCA4.nc")
ffmc_ds = xr.open_dataset("D:/FFMC/FFMC-ICHEC-EC-EARTH-SMHI-RCA4.nc")

W = wind_ds["sfcWind"].values * 3.6  # m/s a km/h
FFMC = ffmc_ds["FFMC"].values

ntime, nlat, nlon = W.shape
ISI_all = np.empty((ntime, nlat, nlon), dtype=np.float32)

# 🔁 Loop por tiempo
for t in range(ntime):
    if t % 100 == 0:
        print(f"Procesando ISI {t}/{ntime}")
    ISI_all[t] = ISI_numpy(W[t], FFMC[t])

# 📦 Dataset final
ds_isi = xr.Dataset(
    {"ISI": (["time", "rlat", "rlon"], ISI_all)},
    coords={
        "time": wind_ds["time"],
        "rlat": wind_ds["rlat"],
        "rlon": wind_ds["rlon"],
        "lat": wind_ds["lat"],
        "lon": wind_ds["lon"]
    }
)

# 💾 Guardar comprimido
output_path = "C:/Users/ferfo/OneDrive/Escritorio/ISI-ICHEC-EC-EARTH-SMHI-RCA4.nc"
ds_isi.to_netcdf(output_path, encoding={"ISI": {"zlib": True, "complevel": 4}})

print(f"✅ ¡ISI calculado y guardado en {output_path}!")

# Cerrar archivos
wind_ds.close()
ffmc_ds.close()
