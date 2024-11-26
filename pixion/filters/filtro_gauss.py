# filtro gaussiano
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import time
import math


# Filtro Gaussiano se manda valores por defecto en caso de que no se especifique
def crear_filtro_gauss(TAM_KERNEL=9):
    desviacion = 2.0
    s = 2.0 * desviacion * desviacion
    suma = 0.0

    mitad = TAM_KERNEL // 2
    filtro_gauss = np.zeros((TAM_KERNEL, TAM_KERNEL), dtype=np.float64)

    # Generación del filtro Gaussiano
    for x in range(-mitad, mitad + 1):
        for y in range(-mitad, mitad + 1):
            r = math.sqrt(x * x + y * y)
            filtro_gauss[x + mitad, y + mitad] = (math.exp(-(r * r) / s)) / (math.pi * s)
            suma += filtro_gauss[x + mitad, y + mitad]

    # Normalización
    filtro_gauss /= suma
    return filtro_gauss

# Conversión a escala de grises
def convertir_a_gris(imagen_original):
    ancho, alto = imagen_original.size
    imagen_gris = np.array(imagen_original.convert('L')).astype(np.uint8)
    return imagen_gris

# Kernel CUDA ajustado
mod = SourceModule("""
    #define TAM_KERNEL 9

    __global__ void aplicar_filtro_gauss_kernel(unsigned char* imagenGris, unsigned char* imagenFiltrada, int ancho, int alto, double* filtroGauss) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int x = tid % ancho;
        int y = tid / ancho;

        if (x >= ancho || y >= alto) return;

        double suma = 0.0;
        int mitad = TAM_KERNEL / 2;

        for (int ky = -mitad; ky <= mitad; ky++) {
            for (int kx = -mitad; kx <= mitad; kx++) {
                int pixelX = max(0, min(ancho - 1, x + kx));
                int pixelY = max(0, min(alto - 1, y + ky));
                suma += imagenGris[pixelY * ancho + pixelX] * filtroGauss[(ky + mitad) * TAM_KERNEL + (kx + mitad)];
            }
        }

        imagenFiltrada[y * ancho + x] = (unsigned char)min(max(suma, 0.0), 255.0);
    }
""")



def apply_gauss(image_path, parametros):
    # funcioon de C
    aplicar_filtro_gauss_kernel = mod.get_function("aplicar_filtro_gauss_kernel")

    
    imagen_original = Image.open(image_path)
    ancho, alto = imagen_original.size

    # Convertir la imagen a escala de grises
    imagen_gris = convertir_a_gris(imagen_original)
    imagen_filtrada = np.zeros_like(imagen_gris, dtype=np.uint8)

    # Crear filtro gaussiano
    filtro_gauss = crear_filtro_gauss(parametros["kernel_size"])
    


    # Asignar memoria en la GPU y copiar datos usando drv.In y drv.Out
    start_time = time.time()

    # Configurar bloques y grid
    total_pixels = ancho * alto
    block_size = parametros["blocks_num"]
    grid_size = (total_pixels + block_size - 1) // block_size
    grid_size_str = str(grid_size)

    aplicar_filtro_gauss_kernel(
        drv.In(imagen_gris), drv.Out(imagen_filtrada), np.int32(ancho), np.int32(alto),
        drv.In(filtro_gauss), block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    drv.Context.synchronize()

    end_time = time.time()

    # calcular tiempo de ejecucion
    execution_time = (end_time - start_time) * 1000  # convertir a milisegundos

     # Formatear tiempo de ejecución
    if execution_time > 1000:
        time_str = f"{execution_time/1000:.2f} segundos"
    else:
        time_str = f"{execution_time:.2f} ms"

    

    # Guardar imagen filtrada
    imagen_filtrada_pil = Image.fromarray(imagen_filtrada)

    return imagen_filtrada_pil, time_str, grid_size_str
    
    