#!/bin/bash

# ===================================================================
# CONFIGURACIÓN REQUERIDA
# ===================================================================

# 1. Directorio raíz donde comienza la búsqueda (donde se encuentra 'output')
#    (Ej: /home/usuario/proyectos)
SOURCE_ROOT="/home/users/cdonoso/astromer/presentation/pipelines/referee"

# 2. Directorio de destino
DESTINATION_ROOT="/home/users/cdonoso/astromer/presentation/pipelines/referee/output_old"

# Patrón de búsqueda relativo a SOURCE_ROOT.
# Busca todas las carpetas 'skip' dentro de 'output/clf*'.
SEARCH_PATTERN="output/clf*_*_500/skip"

# ===================================================================
# LOGICA DEL SCRIPT
# ===================================================================

echo "Buscando carpetas que coincidan con: $SOURCE_ROOT/$SEARCH_PATTERN"
echo "Moviendo a: $DESTINATION_ROOT"
echo "----------------------------------------"

# 1. Usar 'find' para localizar todas las carpetas 'skip' que coincidan con el patrón.
# 2. Usar un bucle while para procesar cada ruta encontrada.
find "$SOURCE_ROOT" -type d -path "$SOURCE_ROOT/$SEARCH_PATTERN" | while read -r SOURCE_DIR; do
    
    # SOURCE_DIR es la ruta completa, ej: /ruta/raiz/output/clf123/skip

    # Extraer la estructura que necesitamos preservar (clf123/skip)
    # 1. Elimina la ruta raíz: /ruta/raiz/output/clf123/skip -> output/clf123/skip
    RELATIVE_PATH="${SOURCE_DIR#$SOURCE_ROOT/}" 
    
    # 2. Elimina 'output/' al inicio: output/clf123/skip -> clf123/skip
    #    (Esto asegura que solo se replique la estructura clf*/skip en el destino)
    STRUCTURE_TO_PRESERVE="${RELATIVE_PATH#output/}"

    # Define el directorio padre de destino (ej: /destino/clf123)
    # Usamos 'dirname' para obtener la ruta sin el '/skip' final.
    TARGET_PARENT_DIR="$DESTINATION_ROOT/$(dirname "$STRUCTURE_TO_PRESERVE")"
    
    # Verifica si la carpeta 'skip' aún existe antes de procesar (es solo una precaución)
    if [ -d "$SOURCE_DIR" ]; then

        # Crear la estructura de directorios de destino (ej: /destino/clf123)
        # La opción -p crea directorios intermedios si no existen.
        mkdir -p "$TARGET_PARENT_DIR"
        
        echo "Moviendo: ${STRUCTURE_TO_PRESERVE}..."
        
        # Mover la carpeta 'skip' completa a la carpeta padre de destino
        # El resultado es /destino/clf123/skip
        mv "$SOURCE_DIR" "$TARGET_PARENT_DIR/"
    fi
done

echo "----------------------------------------"
echo "Proceso de movimiento completado."