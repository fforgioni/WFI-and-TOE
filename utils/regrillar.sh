#!/bin/bash

# Definir rutas de entrada y salida
INPUT_DIR="/mnt/c/Users/ferfo/OneDrive/Escritorio/a"
OUTPUT_DIR="/mnt/c/Users/ferfo/OneDrive/Escritorio/b"
REFERENCE_FILE="/mnt/c/Users/ferfo/OneDrive/Escritorio/cems.nc"  # Archivo de referencia (CEMS)

# Crear la carpeta de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Procesar todos los archivos NetCDF en la carpeta de entrada
for file in "$INPUT_DIR"/*.nc; do
    if [ -f "$file" ]; then
        # Extraer el nombre base del archivo
        filename=$(basename "$file")
        
        # Definir el nombre del archivo de salida
        output_file="$OUTPUT_DIR/$filename"
        
        # Aplicar CDO remapbil para interpolar los datos al sistema de CEMS
        cdo remapbil,"$REFERENCE_FILE" "$file" "$output_file"
        
        echo "Procesado: $filename -> $output_file"
    fi
done

echo "✅ Todos los archivos han sido procesados y guardados en $OUTPUT_DIR"
