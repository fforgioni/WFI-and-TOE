# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 00:06:59 2025

@author: ferfo
"""

import numpy as np
import xarray as xr
import pandas as pd

def FFMC_numpy(TEMP, RH, WIND, RAIN, FFMCPrev):
    RH = np.minimum(RH, 100.0)
    mo = 147.2 * (101.0 - FFMCPrev) / (59.5 + FFMCPrev)

    mo_new = mo.copy()
    mask_rain = RAIN > 0.5

    # 🟦 Lluvia
    if np.any(mask_rain):
        rf = RAIN[mask_rain] - 0.5
        mo_r = mo[mask_rain]
        rf_safe = np.maximum(rf, 0.01)
        exp1 = np.exp(-100.0 / (251.0 - mo_r))
        exp2 = 1.0 - np.exp(-6.93 / rf_safe)
        mr = mo_r + 42.5 * rf * exp1 * exp2
        extra = 0.0015 * (mo_r - 150.0)**2 * np.sqrt(rf)
        mr = np.where(mo_r > 150.0, mr + extra, mr)
        mr = np.minimum(mr, 250.0)
        mo_new[mask_rain] = mr

    # 🟨 Evaporación
    ed = 0.942 * RH**0.679 + 11.0 * np.exp((RH - 100.0) / 10.0) + \
         0.18 * (21.1 - TEMP) * (1.0 - np.exp(-0.115 * RH))

    m = np.empty_like(TEMP)

    # 🟥 mo > ed → secado
    mask_dry = mo_new > ed
    if np.any(mask_dry):
        ko = 0.424 * (1.0 - (RH[mask_dry] / 100.0) ** 1.7) + \
             0.0694 * np.sqrt(WIND[mask_dry]) * (1.0 - (RH[mask_dry] / 100.0) ** 8)
        kd = ko * 0.581 * np.exp(0.0365 * TEMP[mask_dry])
        m[mask_dry] = ed[mask_dry] + (mo_new[mask_dry] - ed[mask_dry]) * 10 ** (-kd)

    # 🟩 mo <= ed → humedecimiento
    mask_wet = ~mask_dry
    if np.any(mask_wet):
        ew = 0.618 * RH[mask_wet] ** 0.753 + 10.0 * np.exp((RH[mask_wet] - 100.0) / 10.0) + \
             0.18 * (21.1 - TEMP[mask_wet]) * (1.0 - np.exp(-0.115 * RH[mask_wet]))
        mask_more_wet = mo_new[mask_wet] < ew
        k1 = 0.424 * (1.0 - ((100.0 - RH[mask_wet]) / 100.0) ** 1.7) + \
             0.0694 * np.sqrt(WIND[mask_wet]) * (1.0 - ((100.0 - RH[mask_wet]) / 100.0) ** 8)
        kw = k1 * 0.581 * np.exp(0.0365 * TEMP[mask_wet])
        m_val = ew - (ew - mo_new[mask_wet]) * 10 ** (-kw)
        m[mask_wet] = np.where(mask_more_wet, m_val, mo_new[mask_wet])

    return 59.5 * (250.0 - m) / (147.2 + m)

# 📂 Cargar datos
temp_ds = xr.open_dataset("E:/temp/tas-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4.nc")
hum_ds = xr.open_dataset("E:/hurs/hurs-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4.nc")
wind_ds = xr.open_dataset("E:/viento/sfcwind-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4.nc")
pp_ds = xr.open_dataset("E:/pp/pr-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4-corregido.nc")

# Extraer y convertir
T = temp_ds["tas"].values - 273.15
RH = hum_ds["hurs"].values
WIND = wind_ds["sfcWind"].values
RAIN = pp_ds["pr"].values * 86400

ntime, nlat, nlon = T.shape
FFMC_prev = np.full((nlat, nlon), 85.0, dtype=np.float32)
FFMC_all = np.empty((ntime, nlat, nlon), dtype=np.float32)

# 🔁 Loop temporal optimizado
for t in range(ntime):
    if t % 100 == 0:
        print(f"🕒 Procesando día {t}/{ntime}")
    FFMC_t = FFMC_numpy(T[t], RH[t], WIND[t], RAIN[t], FFMC_prev)
    FFMC_all[t] = FFMC_t
    FFMC_prev = FFMC_t

# 📦 Dataset xarray
ds_ffmc = xr.Dataset(
    {"FFMC": (["time", "rlat", "rlon"], FFMC_all)},
    coords={
        "time": temp_ds["time"],
        "rlat": temp_ds["rlat"],
        "rlon": temp_ds["rlon"],
        "lat": temp_ds["lat"],
        "lon": temp_ds["lon"]
    }
)

# 💾 Guardar
output_path = "C:/Users/ferfo/OneDrive/Escritorio/FFMC-NOAA-GFDL-GFDL-ESM2M-SMHI-RCA4-OPTIMIZADO.nc"
ds_ffmc.to_netcdf(output_path, encoding={"FFMC": {"zlib": True, "complevel": 4}})

print(f"✅ ¡FFMC calculado y guardado en: {output_path}!")

# Cerrar archivos
temp_ds.close()
hum_ds.close()
wind_ds.close()
pp_ds.close()
