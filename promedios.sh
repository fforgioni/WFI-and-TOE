#!/bin/bash

# Directorios de entrada y salida
INPUT_DIR="/mnt/c/Users/ferfo/OneDrive/Escritorio/a"
OUTPUT_DIR="/mnt/c/Users/ferfo/OneDrive/Escritorio/listo"

# Crear el directorio de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Loop sobre todos los archivos NetCDF en el directorio de entrada
for archivo in "$INPUT_DIR"/*.nc; do
    # Obtener el nombre del archivo sin la ruta
    nombre_archivo=$(basename "$archivo")
    
    # Definir el nombre del archivo de salida en la carpeta de destino
    salida="$OUTPUT_DIR/${nombre_archivo%.nc}_timmean.nc"

    # Aplicar cdo timmean
    cdo timmean "$archivo" "$salida"

    # Mensaje de confirmación
    echo "Procesado: $archivo -> $salida"
done

echo "Proceso completado."
