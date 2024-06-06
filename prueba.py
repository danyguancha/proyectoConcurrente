import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import argparse
import time
import cv2
from tqdm import tqdm


def read_fasta(file_name):
    sequences = []
    for record in SeqIO.parse(file_name, "fasta"):
        sequences.append(str(record.seq))
    return "".join(sequences)


def draw_dotplot(dotplot, fig_name='dotplot.svg'):
    plt.figure(figsize=(10, 10))
    plt.imshow(dotplot, cmap="Greys", aspect="auto")
    plt.xlabel("Secuencia 1")
    plt.ylabel("Secuencia 2")
    plt.savefig(fig_name)
    plt.show()


def dotplot_opencl(sequence1, sequence2):
    # Convert sequences to numpy arrays
    seq1 = np.array(list(sequence1), dtype=np.uint8)
    seq2 = np.array(list(sequence2), dtype=np.uint8)

    # Create OpenCL context and queue
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # OpenCL program
    program = cl.Program(context, """
    __kernel void dotplot(
        __global const uchar *seq1,
        __global const uchar *seq2,
        __global float *dotplot,
        const int len1,
        const int len2)
    {
        int i = get_global_id(0);
        int j = get_global_id(1);

        if (i < len1 && j < len2) {
            if (seq1[i] == seq2[j]) {
                dotplot[i * len2 + j] = (i == j) ? 1.0f : 0.7f;
            } else {
                dotplot[i * len2 + j] = 0.0f;
            }
        }
    }
    """).build()

    # Buffers
    mf = cl.mem_flags
    seq1_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seq1)
    seq2_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seq2)
    dotplot_buf = cl.Buffer(context, mf.WRITE_ONLY, seq1.size * seq2.size * np.dtype(np.float32).itemsize)

    # Execute OpenCL program
    program.dotplot(queue, (seq1.size, seq2.size), None, seq1_buf, seq2_buf, dotplot_buf, np.int32(seq1.size), np.int32(seq2.size))

    # Retrieve result
    dotplot = np.empty(seq1.size * seq2.size, dtype=np.float32)
    cl.enqueue_copy(queue, dotplot, dotplot_buf).wait()

    return dotplot.reshape((seq1.size, seq2.size))


def save_results_to_file(results, file_name="images/results.txt"):
    with open(file_name, "w") as file:
        for result in results:
            file.write(str(result) + "\n")


def acceleration(times):
    return [times[0] / i for i in times]


def efficiency(accelerations, num_threads):
    return [accelerations[i] / num_threads[i] for i in range(len(num_threads))]


def draw_graphic(times, accelerations, efficiencies, num_threads, method):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(num_threads, times)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Tiempo")
    plt.subplot(1, 2, 2)
    plt.plot(num_threads, accelerations)
    plt.plot(num_threads, efficiencies)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Aceleración y Eficiencia")
    plt.legend(["Aceleración", "Eficiencia"])
    plt.savefig(f"images/images_{method}/graficas_{method}.png")


def apply_filter(matrix, path_image):
    kernel_diagonales = np.array([[1, -1, -1],
                                  [-1, 1, -1],
                                  [-1, -1, 1]])

    filtered_matrix = cv2.filter2D(matrix, -1, kernel_diagonales)

    normalized_matrix = cv2.normalize(filtered_matrix, None, 0, 127, cv2.NORM_MINMAX)

    threshold_value = 50
    _, thresholded_matrix = cv2.threshold(normalized_matrix, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imwrite(path_image, thresholded_matrix)
    cv2.imshow('Diagonales', thresholded_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file1', dest='file1', type=str,
                        default=None, help='Query sequence in FASTA format')
    parser.add_argument('--file2', dest='file2', type=str,
                        default=None, help='Subject sequence in FASTA format')

    parser.add_argument('--sequential', action='store_true',
                        help='Ejecutar en modo secuencial')
    parser.add_argument('--opencl', action='store_true',
                        help='Ejecutar utilizando OpenCL')
    args = parser.parse_args()

    file_path_1 = args.file1
    file_path_2 = args.file2

    try:
        merged_sequence_1 = read_fasta(file_path_1)
        merged_sequence_2 = read_fasta(file_path_2)

    except FileNotFoundError as e:
        print("Archivo no encontrado, verifique la ruta")
        exit(1)

    Secuencia1 = merged_sequence_1[0:16000]
    Secuencia2 = merged_sequence_2[0:16000]

    if args.sequential:
        start_secuencial = time.time()
        dotplotSequential = dotplot_sequential(Secuencia1, Secuencia2)
        print(f"Tiempo de ejecución secuencial: {time.time() - start_secuencial}")
        draw_dotplot(dotplotSequential[:600, :600], fig_name="images/images_sequential/dotplot_secuencial.png")
        path_image = 'images/images_filter/dotplot_filter_sequential.png'
        apply_filter(dotplotSequential[:600, :600], path_image)
        save_results_to_file([f"Tiempo de ejecución secuencial: {time.time() - start_secuencial}"], file_name="images/results_sequential.txt")

    if args.opencl:
        start_opencl = time.time()
        dotplotOpenCL = dotplot_opencl(Secuencia1, Secuencia2)
        print(f"Tiempo de ejecución OpenCL: {time.time() - start_opencl}")
        draw_dotplot(dotplotOpenCL[:600, :600], fig_name='images/images_opencl/dotplot_opencl.png')
        path_image = 'images/images_filter/dotplot_filter_opencl.png'
        apply_filter(dotplotOpenCL[:600, :600], path_image)
        save_results_to_file([f"Tiempo de ejecución OpenCL: {time.time() - start_opencl}"], file_name="images/results_opencl.txt")


if __name__ == "__main__":
    main()
