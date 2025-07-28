#!/bin/bash

# Directorio de entrada (donde están los archivos originales)
input_dir="/mnt/c/Users/ferfo/OneDrive/Escritorio/FFMC-listo"

# Directorio de salida (donde se guardarán los resultados)
output_dir="/mnt/c/Users/ferfo/OneDrive/Escritorio/FFMC2"

# Crea el directorio de salida si no existe
mkdir -p "$output_dir"

# Recorre cada archivo NetCDF y aplica cdo monmean
for archivo in "$input_dir"/*.nc; do
    # Obtiene el nombre base del archivo (sin ruta)
    base=$(basename "$archivo")

    # Aplica monmean y guarda en el directorio de salida
    cdo monmean "$archivo" "$output_dir/monmean_$base"

done

# Mensaje final
echo "Proceso terminado. Archivos guardados en $output_dir"
