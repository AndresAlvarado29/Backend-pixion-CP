import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import time

def create_pencil_sketch_kernel(kernel_size):
    # Código del kernel CUDA
    kernel_code = """
    __global__ void applyKernelSketchPencil(unsigned char *input, unsigned char *output, 
                                           int width, int height, int kernelSize)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= width * height) return;
        
        int x = idx % width;
        int y = idx / width;
        
        // Calcular bordes usando Sobel
        float gx = 0.0f, gy = 0.0f;
        int radius = kernelSize / 2;
        
        // Aplicar operador Sobel
        for(int ky = -radius; ky <= radius; ky++) {
            for(int kx = -radius; kx <= radius; kx++) {
                int py = y + ky;
                int px = x + kx;
                
                // Verificar límites
                if(px >= 0 && px < width && py >= 0 && py < height) {
                    float pixel = input[py * width + px];
                    
                    // Kernel Sobel X
                    if(kx == -1) gx -= pixel;
                    if(kx == 1) gx += pixel;
                    
                    // Kernel Sobel Y
                    if(ky == -1) gy -= pixel;
                    if(ky == 1) gy += pixel;
                }
            }
        }
        
        // Magnitud del gradiente
        float magnitude = sqrtf(gx * gx + gy * gy);
        
        // Invertir y ajustar contraste
        float sketch = 255.0f - magnitude;
        sketch = sketch * 1.5f;  // Aumentar contraste
        
        // Saturar valores
        if(sketch > 255.0f) sketch = 255.0f;
        if(sketch < 0.0f) sketch = 0.0f;
        
        output[idx] = (unsigned char)sketch;
    }
    """
    return SourceModule(kernel_code)

def apply_pencil_sketch(image_path, parametros):
    
    kernel_size = parametros["kernel_size"]

    # Leer y preparar la imagen usando Pillow
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
    mod = create_pencil_sketch_kernel(kernel_size)
    sketch_kernel = mod.get_function("applyKernelSketchPencil")
    
    # Configurar bloques y grid
    block_size = parametros["blocks_num"]
    grid_size = (total_pixels + block_size - 1) // block_size
    grid_size_str = str(grid_size)
    # Ejecutar kernel y medir tiempo
    start_time = time.time()
    sketch_kernel(
        input_gpu, output_gpu,
        np.int32(width), np.int32(height),
        np.int32(kernel_size),
        block=(block_size, 1, 1),
        grid=(grid_size, 1)
    )
    cuda.Context.synchronize()
    end_time = time.time()
    
    # calcular tiempo de ejecucion
    execution_time = (end_time - start_time) * 1000  # convertir a milisegundos
    
    # Obtener resultado
    output = np.empty_like(img_array)
    cuda.memcpy_dtoh(output, output_gpu)
    
    # Convertir resultado a imagen Pillow
    result_image = Image.fromarray(output)
    
    # Formatear tiempo de ejecución
    if execution_time > 1000:
        time_str = f"{execution_time/1000:.2f} segundos"
    else:
        time_str = f"{execution_time:.2f} ms"
    
    return result_image, time_str, grid_size_str

# ## Ejecucion

# image_path = "imgCP.jpg"
# kernel_size = 3 
    
# # Procesar imagen
# result, execution_time = apply_pencil_sketch(image_path, kernel_size)
# print(f"Tiempo de ejecución del kernel: {execution_time}")
    
# # Mostrar y guardar resultados
# result.save("pencil_sketch_pythoncpu-3.jpg")


