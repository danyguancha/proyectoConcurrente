from Bio import SeqIO
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse
import time
from mpi4py import MPI
from multiprocessing import Pool
import cv2
from tqdm import tqdm # esta librería es para mirar el progreso de un for


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


def dotplot_sequential(sequence1, sequence2):
    dotplot = np.empty((len(sequence1), len(sequence2)))    
    for i in tqdm(range(len(sequence1))):
        for j in range(len(sequence2)):
            if sequence1[i] == sequence2[j]:
                if i == j:
                    dotplot[i, j] = 1
                else:
                    dotplot[i, j] = 0.7
            else:
                dotplot[i, j] = 0
    return dotplot

def worker_multiprocessing(args):
    i, sequence1, sequence2 = args
    dotplot = []
    for j in range(len(sequence2)):
        if sequence1[i] == sequence2[j]:
            if i == j:
                dotplot.append(1)
            else:
                dotplot.append(0.7)
        else:
            dotplot.append(0)
    return dotplot


def parallel_multiprocessing_dotplot(sequence1, sequence2, threads=mp.cpu_count()):
    with mp.Pool(processes=threads) as pool:
        dotplot = pool.map(worker_multiprocessing, [
                           (i, sequence1, sequence2) for i in range(len(sequence1))])
    return dotplot


def save_results_to_file(results, file_name="images/results.txt"):
    with open(file_name, "w") as file:
        for result in results:
            file.write(str(result) + "\n")


def acceleration(times):
    return [times[0] / i for i in times]


def efficiency(accelerations, num_threads):
    return [accelerations[i] / num_threads[i] for i in range(len(num_threads))]


def draw_graphic_multiprocessing(times, accelerations, efficiencies, num_threads):
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
    plt.savefig("images/images_multiprocessing/graficasMultiprocessing.png")


def draw_graphic_mpi(times, accelerations, efficiencies, num_threads):
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
    plt.savefig("images/images_mpi/graficasMPI.png")

def parallel_mpi_dotplot(sequence_1, sequence_2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunks = np.array_split(range(len(sequence_1)), size)

    dotplot = np.empty([len(chunks[rank]), len(sequence_2)], dtype=np.float16)

    for i in tqdm(range(len(chunks[rank]))):
        for j in range(len(sequence_2)):
            if sequence_1[chunks[rank][i]] == sequence_2[j]:
                if (i == j):
                    dotplot[i, j] = np.float16(1.0)
                else:
                    dotplot[i, j] = np.float16(0.6)
            else:
                dotplot[i, j] = np.float16(0.0)

    dotplot = comm.gather(dotplot, root=0)

    if rank == 0:
        merged_data = np.vstack(dotplot)
        end = time.time()

        return merged_data

def apply_filter(matrix, path_image):
    #matrix = matrix.astype(np.uint8)
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument('--file1', dest='file1', type=str,
                        default=None, help='Query sequence in FASTA format')
    parser.add_argument('--file2', dest='file2', type=str,
                        default=None, help='Subject sequence in FASTA format')

    parser.add_argument('--sequential', action='store_true',
                        help='Ejecutar en modo secuencial')
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Ejecutar utilizando multiprocessing')
    parser.add_argument('--mpi', action='store_true',
                        help='Ejecutar utilizando mpi4py')
    parser.add_argument('--num_processes', dest='num_processes', type=int, nargs='+',
                        default=[4], help='Número de procesos para la opción MPI')
    args = parser.parse_args()

    if rank == 0:
        chargeFilesStart= time.time();
        file_path_1 = args.file1
        file_path_2 = args.file2

        num_threads_array = args.num_processes

        try:
            merged_sequence_1 = read_fasta(file_path_1)
            merged_sequence_2 = read_fasta(file_path_2)

        except FileNotFoundError as e:
            print("Archivo no encontrado, verifique la ruta")
            exit(1)

        Secuencia1 = merged_sequence_1[0:16000]
        Secuencia2 = merged_sequence_2[0:16000]
        chargeFilesFinish=time.time()

        save_results_to_file([F"Tiempo de carga de los archivos: {chargeFilesFinish - chargeFilesStart}"],file_name="images/results_charge_files.txt")

        dotplot = np.empty([len(Secuencia1), len(Secuencia2)])
        results_print = []
        results_print_mpi = []
        times_multiprocessing = []
        times_mpi = []

    if args.sequential:
        start_secuencial = time.time()
        dotplotSequential = dotplot_sequential(Secuencia1, Secuencia2)
        results_print.append(
            f"Tiempo de ejecución secuencial: {time.time() - start_secuencial}")
        draw_dotplot(dotplotSequential[:600, :600], fig_name="images/images_sequential/dotplot_secuencial.png")
        path_image = 'images/images_filter/dotplot_filter_sequential.png'  
        apply_filter(dotplotSequential[:600, :600], path_image)
        save_results_to_file(results_print,file_name="images/results_sequential.txt")

    if args.multiprocessing:
        num_threads = [1, 2, 4, 8]
        for num_thread in num_threads:
            start_time = time.time()
            dotplotMultiprocessing = np.array(
                parallel_multiprocessing_dotplot(Secuencia1, Secuencia2, num_thread))
            times_multiprocessing.append(time.time() - start_time)
            results_print.append(
                f"Tiempo de ejecución multiprocessing con {num_thread} hilos: {time.time() - start_time}")
            
        # Aceleración
        accelerations = acceleration(times_multiprocessing)
        for i in range(len(accelerations)):
            results_print.append(
                f"Aceleración con {num_threads[i]} hilos: {accelerations[i]}")

        # Eficiencia
        efficiencies = efficiency(accelerations, num_threads)
        for i in range(len(efficiencies)):
            results_print.append(
                f"Eficiencia con {num_threads[i]} hilos: {efficiencies[i]}")

        save_results_to_file(results_print,file_name="images/results_multiprocessing.txt")
        draw_graphic_multiprocessing(
            times_multiprocessing, accelerations, efficiencies, num_threads)
        draw_dotplot(dotplotMultiprocessing[:600, :600],
                     fig_name='images/images_multiprocessing/dotplot_multiprocessing.png')
        
        path_image = 'images/images_filter/dotplot_filter_multiprocessing.png'  
        apply_filter(dotplotMultiprocessing[:600, :600], path_image)

    if args.mpi:
        for thread in num_threads_array:
            start_time = time.time()
            dotplot_mpi = parallel_mpi_dotplot(Secuencia1, Secuencia2)
            times_mpi.append(time.time() - start_time)
            results_print_mpi.append(
                f"Tiempo de ejecución mpi con {thread} hilos: {time.time() - start_time}")
            
        accelerations = acceleration(times_mpi)
        for i in range(len(accelerations)):
            results_print_mpi.append(
                f"Aceleración con {num_threads_array[i]} hilos: {accelerations[i]}")
        
        efficiencies = efficiency(accelerations, num_threads_array)
        for i in range(len(efficiencies)):
            results_print_mpi.append(
                f"Eficiencia con {num_threads_array[i]} hilos: {efficiencies[i]}")

        save_results_to_file(results_print_mpi,file_name="images/results_mpi.txt")
        draw_graphic_mpi(
            times_mpi, accelerations, efficiencies, num_threads_array)
        draw_dotplot(dotplot_mpi[:600, :600],
                     fig_name='images/images_mpi/dotplot_mpi.png')
        
        path_image = 'images/images_filter/dotplot_filter_mpi.png'  
        apply_filter(dotplot_mpi[:600, :600], path_image)
        


if __name__ == "__main__":
    main()
