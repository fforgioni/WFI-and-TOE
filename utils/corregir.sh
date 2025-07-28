#!/bin/bash

# Definir directorios
INPUT_DIR="/mnt/c/Users/ferfo/OneDrive/Escritorio/DC"
OUTPUT_DIR="/mnt/c/Users/ferfo/OneDrive/Escritorio/DC1"

# Crear la carpeta de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Procesar todos los archivos NetCDF en la carpeta de entrada
for file in "$INPUT_DIR"/*.nc; do
    if [ -f "$file" ]; then
        # Extraer el nombre base del archivo
        filename=$(basename "$file")
        
        # Definir el nombre del archivo de salida
        output_file="$OUTPUT_DIR/$filename"
        
        # Aplicar CDO para limitar valores a un máximo de 150
        cdo setrtoc,800,1e9,800 "$file" "$output_file"
        
        echo "Procesado: $filename -> $output_file"
    fi
done

echo "Todos los archivos han sido procesados y guardados en $OUTPUT_DIR"
