# filtro de erosion
import numpy as np
import time
from PIL import Image
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time

# Conversión a escala de grises
def convertir_a_gris(imagen_original):
    ancho, alto = imagen_original.size
    imagen_gris = np.array(imagen_original.convert('L')).astype(np.uint8)
    return imagen_gris


mod = SourceModule("""
    #define TAM_KERNEL 9  // Tamaño del kernel de erosión

    __global__ void aplicar_filtro_erosion_kernel(unsigned char* imagenGris, unsigned char* imagenFiltrada, int ancho, int alto) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int x = tid % ancho;
        int y = tid / ancho;

        if (x >= ancho || y >= alto) return;

        unsigned char min_val = 255;  // Máximo valor para escala de grises
        int mitad = TAM_KERNEL / 2;

        for (int ky = -mitad; ky <= mitad; ky++) {
            for (int kx = -mitad; kx <= mitad; kx++) {
                int pixelX = max(0, min(ancho - 1, x + kx));
                int pixelY = max(0, min(alto - 1, y + ky));
                min_val = min(min_val, imagenGris[pixelY * ancho + pixelX]);
            }
        }

        imagenFiltrada[y * ancho + x] = min_val;
    }
""")

def apply_erosion(image_path, parametros):
    aplicar_filtro_erosion_kernel = mod.get_function("aplicar_filtro_erosion_kernel")

    
    imagen_original = Image.open(image_path)
    ancho, alto = imagen_original.size

    # Convertir la imagen a escala de grises
    imagen_gris = convertir_a_gris(imagen_original)
    imagen_filtrada = np.zeros_like(imagen_gris, dtype=np.uint8)

    # Configurar bloques y grid
    block_size = parametros["blocks_num"]
    grid_size = ((ancho*alto) + block_size - 1) // block_size
    grid_size_str = str(grid_size)
    

    # Asignar memoria en la GPU y copiar datos usando drv.In y drv.Out
    start = time.time()

    aplicar_filtro_erosion_kernel(
        drv.In(imagen_gris), drv.Out(imagen_filtrada), np.int32(ancho), np.int32(alto),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    drv.Context.synchronize()

    end = time.time()

    # calcular tiempo de ejecucion
    execution_time = (end - start) * 1000 

    
    imagen_filtrada_pil = Image.fromarray(imagen_filtrada)
    
    # Formatear tiempo de ejecución
    if execution_time > 1000:
        time_str = f"{execution_time/1000:.2f} segundos"
    else:
        time_str = f"{execution_time:.2f} ms"
    
    return imagen_filtrada_pil, time_str, grid_size_str