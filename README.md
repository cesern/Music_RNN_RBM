# Generando música de piano con RNN-RBM
Generador de música polifonica con RNN-RBM en TensorFlow, fork de [Music_RNN_RBM](https://github.com/cesern/Music_RNN_RBM) de [Dan Shiebler](https://github.com/dshieble).

##  Las RNN
Una red neuronal recurrente es una arquitectura de red neuronal que puede manejar secuencias de vectores. Esto lo hace perfecto para trabajar con datos temporales. Por ejemplo, los RNN son excelentes en tareas como predecir la siguiente palabra en una oración o pronosticar una serie de tiempo.

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
