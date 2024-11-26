import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import time

# Código del kernel CUDA para el filtro negativo personalizado
def create_negativo_kernel(kernel_size):
    kernel_code = f"""
    __global__ void aplicarNegativoPersonalizado(unsigned char *input, unsigned char *output, 
                                                 int width, int height, int kernel_size, int intensidad)
    {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= width * height) return;

        int x = idx % width;
        int y = idx / width;

        int radius = kernel_size / 2;
        int offset = kernel_size * kernel_size;

        float pixel_value = 0.0f;
        for (int ky = -radius; ky <= radius; ky++) {{
            for (int kx = -radius; kx <= radius; kx++) {{
                int px = x + kx;
                int py = y + ky;

                if (px >= 0 && px < width && py >= 0 && py < height) {{
                    int index = (py * width + px);
                    if (kx == -ky || kx == ky) {{
                        pixel_value += intensidad - input[index];
                    }}
                }}
            }}
        }}

        int center_idx = y * width + x;
        output[center_idx] = (unsigned char)fminf(fmaxf(pixel_value, 0.0f), 255.0f);
    }}
    """
    return SourceModule(kernel_code)

# Función para aplicar el filtro negativo personalizado
def apply_negativo_personalizado(image_path, kernel_size=3, intensidad=40):
    # Leer y preparar la imagen
    img = Image.open(image_path).convert('L')  # Convertir a escala de grises
    width, height = img.size
    total_pixels = width * height

    # Convertir imagen a array de numpy
    img_array = np.array(img, dtype=np.uint8)

    # Preparar datos para GPU
    input_gpu = cuda.mem_alloc(img_array.nbytes)
    output_gpu = cuda.mem_alloc(img_array.nbytes)

    # Copiar imagen a GPU
    cuda.memcpy_htod(input_gpu, img_array)

    # Compilar y obtener función del kernel
    mod = create_negativo_kernel(kernel_size)
    negativo_kernel = mod.get_function("aplicarNegativoPersonalizado")

    # Configurar bloques y grid
    block_size = 1024
    grid_size = (total_pixels + block_size - 1) // block_size

    # Ejecutar kernel y medir tiempo
    start_time = time.time()
    negativo_kernel(
        input_gpu, output_gpu,
        np.int32(width), np.int32(height),
        np.int32(kernel_size), np.int32(intensidad),
        block=(block_size, 1, 1),
        grid=(grid_size, 1)
    )
    cuda.Context.synchronize()
    end_time = time.time()

    # Calcular tiempo de ejecución
    execution_time = (end_time - start_time) * 1000  # convertir a milisegundos

    # Obtener resultado
    output = np.empty_like(img_array)
    cuda.memcpy_dtoh(output, output_gpu)

    # Convertir resultado a imagen Pillow
    result_image = Image.fromarray(output)

    # Formatear tiempo de ejecución
    if execution_time > 1000:
        time_str = f"{execution_time / 1000:.2f} segundos"
    else:
        time_str = f"{execution_time:.2f} ms"

    return result_image, time_str

# # Ejecución
# image_path = "/home/andres/ambientes virtuales/PruebasML/Practicas/CP/PracticaPyCuda/NuevosFiltros/e93438c5c77eaecf81afe66f26892068_bp.png"  # Cambiar por la ruta de tu imagen
# kernel_size = 49
# intensidad = 10

# # Procesar la imagen
# result, execution_time = apply_negativo_personalizado(image_path, kernel_size, intensidad)
# print(f"Tiempo de ejecución del kernel: {execution_time}")

# # Guardar el resultado
# result.save("resultado_negativo_cuda.png")
