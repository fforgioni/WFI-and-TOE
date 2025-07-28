# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 00:22:14 2025

@author: ferfo
"""

import numpy as np
import xarray as xr

def FWI_numpy(ISI, BUI):
    """Versión vectorizada del FWI."""
    fD = np.where(
        BUI <= 80.0,
        0.626 * np.power(BUI, 0.809) + 2.0,
        1000.0 / (25.0 + 108.64 * np.exp(-0.023 * BUI))
    )

    B = 0.1 * ISI * fD

    S = np.where(
        B > 1.0,
        np.exp(2.72 * np.power(0.434 * np.log(np.maximum(B, 1e-10)), 0.647)),
        B
    )

    return S

# 📂 Cargar archivos
isi_ds = xr.open_dataset("D:ISI/ISI-CCCma-CanESM2-SMHI-RCA4.nc")
bui_ds = xr.open_dataset("D:BUI/BUI-CCCma-CanESM2-SMHI-RCA4.nc")

ISI_vals = isi_ds["ISI"].values
BUI_vals = bui_ds["BUI"].values
ntime, nlat, nlon = ISI_vals.shape

FWI_all = np.empty((ntime, nlat, nlon), dtype=np.float32)

# 🔁 Loop temporal
for t in range(ntime):
    if t % 100 == 0:
        print(f"Calculando FWI para t={t}/{ntime}")
    FWI_all[t] = FWI_numpy(ISI_vals[t], BUI_vals[t])

# 📦 Crear dataset final
ds_fwi = xr.Dataset(
    {"FWI": (["time", "rlat", "rlon"], FWI_all)},
    coords={
        "time": isi_ds["time"],
        "rlat": isi_ds["rlat"],
        "rlon": isi_ds["rlon"],
        "lat": isi_ds["lat"],
        "lon": isi_ds["lon"]
    }
)

# 💾 Guardar con compresión
output_file = "C:/Users/ferfo/OneDrive/Escritorio/FWI-CCCma-CanESM2-SMHI-RCA4.nc"
ds_fwi.to_netcdf(output_file, encoding={"FWI": {"zlib": True, "complevel": 4}})
print(f"✅ ¡FWI calculado y guardado en {output_file}!")

# 📁 Cerrar datasets
isi_ds.close()
bui_ds.close()
