# Generando música de piano con RNN-RBM
Generador de música polifonica con RNN-RBM en TensorFlow, fork de [Music_RNN_RBM](https://github.com/cesern/Music_RNN_RBM) de [Dan Shiebler](https://github.com/dshieble).

##  Las RNN
Una red neuronal recurrente es una arquitectura de red neuronal que puede manejar secuencias de vectores. Esto lo hace perfecto para trabajar con datos temporales. Por ejemplo, los RNN son excelentes en tareas como predecir la siguiente palabra en una oración o pronosticar una serie de tiempo.

Una RNN es una secuencia de unidades de red neuronal donde cada unidad toma información tanto de la unidad anterior como del vector de datos y produce una salida. Hoy en día, la mayoría de las redes neuronales recurrentes utilizan una arquitectura llamada memoria de corto plazo a largo plazo (LSTM).
## RBM
RBM es una red neuronal con 2 capas, la capa visible y la capa oculta. Cada nodo visible está conectado a cada nodo oculto (y viceversa), pero no hay conexiones visibles-visibles u ocultas-ocultas. Los parámetros de la RBM son la matriz de ponderación W y los vectores de polarización bh y bv.

![rbm](images/RBM.png)

Una RBM es capaz de modelar una distribución de probabilidad dimensional alta y compleja a partir de la cual podemos generar notas musicales.
## Las RNN-RBM
La RNN-RBM es un modelo generativo no supervisado. Esto significa que el objetivo del algoritmo es modelar directamente la distribución de probabilidad de un conjunto de datos sin etiquetar. Podemos pensar en el RNN-RBM como una serie de máquinas Boltzmann restringidas cuyos parámetros están determinados por un RNN.

En el RNN-RBM, las unidades ocultas de RNN comunican información sobre el acorde que se está reproduciendo o el estado de ánimo de la canción a los RBM. Esta información condiciona las distribuciones de probabilidad de RBM, para que la red pueda modelar la forma en que las notas cambian a lo largo de una canción.

La arquitectura de la RNN-RBM no es tremendamente complicada. Cada unidad oculta de RNN se empareja con un RBM. La unidad oculta RNN toma la entrada del vector de datos (la nota o notas en un tiempo t), así como la unidad oculta RNN anterior. Las salidas de la unidad oculta son los parámetros de la RBM siguiente, que toma como vector de datos de entrada `v` en el tiempo siguiente.

![rnn](images/RNNRBM.png)

Todos los RBMs comparten la misma matriz de pesos, y solo los vectores bias ocultos y visibles están determinados por las salidas de las unidades RNN. Con esta convención, el rol de la matriz de peso RBM es especificar un previo consistente en todas las distribuciones RBM (la estructura general de la música), y el rol de los vectores bias es comunicar información temporal (el estado actual de la canción).

## Entrenamiento

1. En el primer paso, comunicamos los conocimientos de RNN sobre el estado actual de la canción a la RBM
2. En el segundo paso creamos unas pocas notas musicales con el RBM.
3. En el tercer paso, comparamos las notas que generamos con las notas reales de la canción en el tiempo t. Luego, propagamos esta pérdida a través de la red para calcular los gradientes de los parámetros de la red y actualizar los pesos y sesgos de la red. En la práctica, TensorFlow manejará los pasos de backpropagation y update.
4. En el cuarto paso, utilizamos la nueva información para actualizar la representación interna de RNN del estado de la canción.

## Datos

El formato de datos es archivos de piano midi. Para el entrenamiento use 173 archivos midi de Johann Sebastian Bach ([Descargar](http://www.bachcentral.com/midiindexcomplete.html)).

## Resultados

* [Resultado 1 con 100 epoch](results/0_resultado_100.mid)
* [Resultado 2 con 100 epoch](results/1_resultado_100.mid)
* [Resultado 3 con 100 epoch](results/2_resultado_100.mid)
* [Resultado 1 con 500 epoch](results/0_resultado_500.mid)
* [Resultado 2 con 500 epoch](results/1_resultado_500.mid)
* [Resultado 3 con 500 epoch](results/2_resultado_500.mid)

## Conclusión

En general fue un proyecto sencillo de entender, de entrenar, y generar la música. El tiempo de entrenamiento fue relativamente rápido, con 173 muestras de archivos midi y 500 epoch el tiempo fue alrededor de 2 horas y la generación de la musica fue tambien rápida.

Algo a comentar es que influyen mucho los datos de entrenamiento y la cantidad porque entrene con unos pocos y distintos archivos midi y algunos resultados eran muy malos. Cuando entrené con el conjunto de _Bash_ obtuve los mejores resultados de todos los que entrené.

## Uso
### Entrenamiento
Para entrenar el modelo, primero ejecutar el siguiente comando para iniciar los parametros de la RBM.
```
python weight_initializations.py
```
Luego, ejecutar lo siguiente para entrenar el modelo RNN_RBM:
```
python rnn_rbm_train.py <num_epochs>
```
### Generación:
Correr el comando siguiente, con el se generará musica. 
```
python rnn_rbm_generate.py <path_to_ckpt_file>
```
Cuando ejecutas `train_rnn_rbm.py`, se crea `epoch_<x>.ckpt` en el directorio _parameter_checkpoints_. Comando de ejemplo: `python rnn_rbm_generate.py epoch_100.ckpt`.

___
