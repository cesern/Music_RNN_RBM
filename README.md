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

Todos los RBMs comparten la misma matriz de pesos, y solo los vectores bias ocultos y visibles están determinados por las salidas de las unidades RNN. Con esta convención, el rol de la matriz de peso RBM es especificar un previo consistente en todas las distribuciones RBM (la estructura general de la música), y el rol de los vectores bias es comunicar información temporal (el estado actual de la canción).
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
