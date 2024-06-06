# Proyecto Final:

## Análisis de Rendimiento de Dotplot

## Secuencial vs Paralelización

El objetivo de este proyecto es implementar y analizar el rendimiento de tres formas de realizar un dotplot, una técnica comúnmente utilizada en bioinformática para comparar secuencias de ADN o proteínas.

### Prerequisitos

El proyecto fue desarrollado usando Python 3.10.9 y con soporte de computación paralela usando librerias multiprocessing y mpi4py. Requiere parámetros de entrada como la secuencia de referencia y de consulta en formato fna que deben declararse en la línea de comandos de ejecución para calcular el dot-plot.

### Instalacion

Tener instalado python, para la posterior instalación de la slibrerias necesarias

A continuación, instale los paquetes de Python necesarios

```
pip install numpy
pip install matplotlib
pip install mpi4py
pip install biopython
pip install opencv-python
pip install tqdm==2.2.3
```

Finalmente, descargue el repositorio

```
git clone https://github.com/cristianHenao00/PCD-project.git
```

### Ejecución

Para ejecutar el programa secuencial, ejecute el siguiente comando:

```
python proyecto.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --sequential
```

Para ejecutar multiprocessing, ejecute el siguiente comando:

```
python proyecto.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --multiprocessing
```

Para ejecutar mpi4py, ejecute el siguiente comando:

```
python proyecto.py --num_processes 1 2 3 4 --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --mpi
```
