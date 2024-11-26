# filtro ups :

import numpy as np
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Ruta del logotipo
LOGO_PATH = ".logoups.jpeg"

def apply_mosaic_logo_filter(input_image_path, output_image_path, logo_path=LOGO_PATH, transparency=0.5):
    """
    Aplica un filtro de mosaico con un logotipo en la imagen de entrada.

    Args:
        input_image_path (str): Ruta de la imagen de entrada.
        output_image_path (str): Ruta de la imagen de salida.
        logo_path (str): Ruta del logotipo.
        transparency (float): Nivel de transparencia del logotipo (0.0 a 1.0).
    """
    # Cargar la imagen principal y el logotipo
    image = Image.open(input_image_path).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")

    # Redimensionar el logotipo a un tamaño manejable
    logo = logo.resize((100, 100), Image.Resampling.LANCZOS)

    # Convertir imágenes a matrices numpy
    img_array = np.array(image)
    logo_array = np.array(logo)

    # Obtener dimensiones
    img_h, img_w, _ = img_array.shape
    logo_h, logo_w, _ = logo_array.shape

    # Calcular el número de repeticiones en mosaico
    num_tiles_x = img_w // logo_w
    num_tiles_y = img_h // logo_h

    # Crear un mosaico del logotipo
    mosaic_logo = np.zeros_like(img_array)
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            mosaic_logo[y * logo_h:(y + 1) * logo_h, x * logo_w:(x + 1) * logo_w, :] = logo_array

    # Cargar datos en la GPU
    img_gpu = cuda.mem_alloc(img_array.nbytes)
    mosaic_gpu = cuda.mem_alloc(mosaic_logo.nbytes)
    cuda.memcpy_htod(img_gpu, img_array)
    cuda.memcpy_htod(mosaic_gpu, mosaic_logo)

    # Código CUDA para combinar imágenes con transparencia
    mod = SourceModule("""
    _global_ void blend_mosaic(
        unsigned char *image, unsigned char *mosaic,
        int img_w, int img_h, float transparency) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= img_w || y >= img_h) return;

        int idx = (y * img_w + x) * 4;

        for (int i = 0; i < 3; ++i) {
            image[idx + i] = (unsigned char)(
                transparency * mosaic[idx + i] + (1.0 - transparency) * image[idx + i]
            );
        }
    }
    """)

    blend_mosaic = mod.get_function("blend_mosaic")

    # Configurar los bloques e hilos
    block_size = (16, 16, 1)
    grid_size = ((img_w + block_size[0] - 1) // block_size[0],
                 (img_h + block_size[1] - 1) // block_size[1], 1)

    # Ejecutar el kernel
    blend_mosaic(img_gpu, mosaic_gpu,
                 np.int32(img_w), np.int32(img_h),
                 np.float32(transparency),
                 block=block_size, grid=grid_size)

    # Descargar los datos procesados de la GPU
    cuda.memcpy_dtoh(img_array, img_gpu)

    # Guardar la imagen resultante
    result_image = Image.fromarray(img_array, "RGBA")
    result_image.save(output_image_path)
    print(f"Filtro de mosaico aplicado y guardado en {output_image_path}")
    return result_image

# Ejemplo de uso
#apply_mosaic_logo_filter("/home/andres/ambientes virtuales/PruebasML/Practicas/CP/PracticaPyCuda/NuevosFiltros/e93438c5c77eaecf81afe66f26892068_bp.png", "output_mosaic_image.png")