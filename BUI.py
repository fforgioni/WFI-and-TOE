# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 00:13:11 2025

@author: ferfo
"""

import numpy as np
import xarray as xr

def BUI_numpy(DMC, DC):
    """Versión vectorizada del BUI."""
    mask = DMC <= 0.4 * DC
    U = np.empty_like(DMC)

    # Caso 1
    U[mask] = 0.8 * DMC[mask] * DC[mask] / np.maximum(DMC[mask] + 0.4 * DC[mask], 0.0001)

    # Caso 2
    denom = DMC[~mask] + 0.4 * DC[~mask]
    parte1 = (1.0 - 0.8 * DC[~mask] / denom)
    parte2 = 0.92 + (0.0114 * DMC[~mask]) ** 1.7
    U[~mask] = DMC[~mask] - parte1 * parte2

    return np.maximum(U, 0.0)

# 📂 Cargar archivos
dmc_ds = xr.open_dataset("D:/DMC/DMC-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4.nc")
dc_ds = xr.open_dataset("D:/DC/DC-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4.nc")

DMC_vals = dmc_ds["DMC"].values
DC_vals = dc_ds["DC"].values
ntime, nlat, nlon = DMC_vals.shape

# 🧱 Array de salida
BUI_all = np.empty((ntime, nlat, nlon), dtype=np.float32)

# 🔁 Loop temporal
for t in range(ntime):
    if t % 100 == 0:
        print(f"Calculando BUI para t={t}/{ntime}")
    BUI_all[t] = BUI_numpy(DMC_vals[t], DC_vals[t])

# 📦 Dataset final
ds_bui = xr.Dataset(
    {"BUI": (["time", "rlat", "rlon"], BUI_all)},
    coords={
        "time": dmc_ds["time"],
        "rlat": dmc_ds["rlat"],
        "rlon": dmc_ds["rlon"],
        "lat": dmc_ds["lat"],
        "lon": dmc_ds["lon"]
    }
)

# 💾 Guardar con compresión
output_file = "C:/Users/ferfo/OneDrive/Escritorio/BUI-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4.nc"
ds_bui.to_netcdf(output_file, encoding={"BUI": {"zlib": True, "complevel": 4}})
print(f"✅ ¡BUI calculado y guardado en {output_file}!")

# 🧹 Cerrar archivos
dmc_ds.close()
dc_ds.close()
