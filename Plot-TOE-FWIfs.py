# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 02:30:06 2025

@author: ferfo
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec

# === Estilo cargado desde diccionario ===
def cargar_estilo_paper():
    estilo_paper = {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#222222",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "hatch.color": "#222222",
        "hatch.linewidth": 1.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
    plt.rcParams.update(estilo_paper)

cargar_estilo_paper()

# === Colormap violeta personalizado ===
colors_toe = ["#c994c7", "#756bb1", "#54278f", "#2a115f"]
cmap_toe = LinearSegmentedColormap.from_list("custom_toe", colors_toe, N=256)

# === Funciones ===
def calcular_anomalia(tas):
    tas = tas.assign_coords(year=tas["time"].dt.year)
    tas_anual = tas.groupby("year").mean(dim=["time", "latitude", "longitude"], skipna=True)
    clim = tas_anual.sel(year=slice(1981, 2010)).mean()
    return tas_anual - clim

def calcular_histograma(toe):
    toe_valid = toe.values[~np.isnan(toe.values)]
    toe_valid = toe_valid[(toe_valid >= 2010) & (toe_valid <= 2070)]
    years = np.arange(2010, 2071)
    hist, _ = np.histogram(toe_valid, bins=years)
    hist_percent = 100 * hist / len(toe_valid)
    cum_percent = np.cumsum(hist_percent)
    year_50 = years[:-1][np.argmax(cum_percent >= 50)]
    return years, hist_percent, year_50

def panel_mapa(ax, data, title, cmap, norm, hatch_mask=None):
    im = ax.pcolormesh(data.longitude, data.latitude, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    if hatch_mask is not None:
        masked = xr.where(hatch_mask, 1, np.nan)
        ax.contourf(data.longitude, data.latitude, masked,
                    levels=[0.5, 1.5], hatches=['///'], colors='none',
                    transform=ccrs.PlateCarree(), zorder=5)

    if "(e)" in title:
        ax.text(0.98, 0.98, "(e)", transform=ax.transAxes,
                fontsize=30, fontweight='bold', va='top', ha='right', zorder =30)
        ax.set_title("FWIfs–RCP4.5", fontsize=18, pad=12)
    elif "(f)" in title:
        ax.text(0.98, 0.98, "(f)", transform=ax.transAxes,
                fontsize=30, fontweight='bold', va='top', ha='right', zorder =30)
        ax.set_title("FWIfs–RCP8.5", fontsize=18, pad=12)


    ax.set_extent([-85, -30, -60, 15], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=-1)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=12)
    return im

def panel_inferior(ax, df, hist, years, year_50, titulo):
    df["mean"] = df["anom"].rolling(window=5, center=True, min_periods=1).mean()
    df["std"] = df["anom"].rolling(window=5, center=True, min_periods=1).std()
    ax.fill_between(df["year"], df["mean"] - df["std"], df["mean"] + df["std"], color="#d73027", alpha=0.3)
    ax.plot(df["year"], df["mean"], color="#d73027", linewidth=2.5)
    ax.set_ylabel("Temp anomaly (°C)", fontsize=22)
    ax.set_xlabel("Year", fontsize=18)

    if "(f)" in titulo:
        ax.text(0.01, 0.98, "(f)", transform=ax.transAxes,
                fontsize=30, fontweight='bold', va='top', ha='left', zorder=20)
        ax.set_title("FWIfs–RCP4.5", fontsize=18, pad=10)
    elif "(e)" in titulo:
        ax.text(0.01, 0.98, "(e)", transform=ax.transAxes,
                fontsize=30, fontweight='bold', va='top', ha='left', zorder=20)
        ax.set_title("FWIfs–RCP8.5", fontsize=18, pad=10)

    ax.set_xlim(2010, 2100)
    ax.set_ylim(0, 3.5)
    ax.set_xticks(np.arange(2010, 2101, 10))
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(labelsize=20)  # <<< tamaño ticks del eje izquierdo e inferior
    ax.grid(False)

    ax2 = ax.twinx()
    ax2.bar(years[:-1], hist, width=0.8, color="#f4a582", edgecolor="black", linewidth=0.5, alpha=0.75)
    ax2.set_ylabel("Percent of emergence (%)", fontsize=22, labelpad=12)
    ax2.set_ylim(0, 20)
    ax2.set_yticks(np.arange(0, 21, 2))
    ax2.tick_params(labelsize=20)
    
    # Línea vertical y etiqueta del año 50%
    ax2.axvline(year_50, color="#8e44ad", linestyle="--", linewidth=2)
    ax2.text(year_50 + 1, ax2.get_ylim()[1] * 0.9,
             f"Year {year_50}", color="#8e44ad", fontsize=14,
             fontweight='bold', va='center', ha='left')

# === Funciones combinadas ===
def plot_doble_mapa(data1, data2, titulo1, titulo2, cmap, norm, bounds, nombre_archivo):
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2, wspace=-0.4)  # menos espacio entre mapas
    proj = ccrs.PlateCarree()

    ax1 = fig.add_subplot(gs[0], projection=proj)
    im1 = panel_mapa(ax1, data1, titulo1, cmap, norm, hatch_mask=(data1 < 2010))

    ax2 = fig.add_subplot(gs[1], projection=proj)
    im2 = panel_mapa(ax2, data2, titulo2, cmap, norm, hatch_mask=(data2 < 2010))

    # Barra de color
    cbar_ax = fig.add_axes([0.80, 0.10, 0.02, 0.8])
    cbar = fig.colorbar(im2, cax=cbar_ax, boundaries=bounds, spacing='proportional')
    cbar.set_label("Year", fontsize=20, labelpad=12)
    cbar.ax.tick_params(labelsize=18)

    # Ajustar layout dejando lugar arriba para el título
    plt.tight_layout(rect=[0, 0, 0.91, 0.94])
    plt.savefig(nombre_archivo, dpi=300)
    plt.show()

def plot_doble_histograma(df1, hist1, years1, y50_1, titulo1,
                          df2, hist2, years2, y50_2, titulo2,
                          nombre_archivo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    panel_inferior(ax1, df1, hist1, years1, y50_1, titulo1)
    panel_inferior(ax2, df2, hist2, years2, y50_2, titulo2)

    # Agregar espacio horizontal entre los gráficos
    plt.subplots_adjust(wspace=1.5)  # <-- probá con 0.15 o 0.20 si querés más

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(nombre_archivo, dpi=300)
    plt.show()

# === Cargar datos ===
toe_45 = xr.open_dataarray("C:/Users/ferfo/OneDrive/Escritorio/ToE_FWIfs_rcp45.nc")
toe_85 = xr.open_dataarray("C:/Users/ferfo/OneDrive/Escritorio/ToE_FWIfs_rcp85.nc")
tas_45 = xr.open_dataset("C:/Users/ferfo/OneDrive/Escritorio/temp-1951-2100-4.5.nc")["tas"]
tas_85 = xr.open_dataset("C:/Users/ferfo/OneDrive/Escritorio/temp-1951-2100-8.5.nc")["tas"]

# === Filtrar valores post-2070 ===
toe_45 = toe_45.where(toe_45 <= 2070)
toe_85 = toe_85.where(toe_85 <= 2070)

# === Procesar datos ===
anom_45 = calcular_anomalia(tas_45)
anom_85 = calcular_anomalia(tas_85)
years_45, hist_45, y50_45 = calcular_histograma(toe_45)
years_85, hist_85, y50_85 = calcular_histograma(toe_85)
bounds_toe = np.arange(2010, 2080, 10)
norm_toe = BoundaryNorm(boundaries=bounds_toe, ncolors=256)


df_45 = pd.DataFrame({"year": anom_45.year.values, "anom": anom_45.values})
df_85 = pd.DataFrame({"year": anom_85.year.values, "anom": anom_85.values})

# === Ejecutar figuras ===
plot_doble_mapa(
    toe_45, toe_85,
    "(e) ToE FWI95d - RCP4.5",
    "(f) ToE FWI95d - RCP8.5",
    cmap_toe, norm_toe, bounds_toe,
    "C:/Users/ferfo/OneDrive/Escritorio/figura_doble_mapa_FWIfs.png"
)

plot_doble_histograma(
    df_45, hist_45, years_45, y50_45, "(e) Temp anomaly + Emergence FWI95d - RCP4.5",
    df_85, hist_85, years_85, y50_85, "(f) Temp anomaly + Emergence FWI95d - RCP8.5",
    "C:/Users/ferfo/OneDrive/Escritorio/figura_doble_histograma_FWIfs.png"
)
