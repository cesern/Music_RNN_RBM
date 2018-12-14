# Generando música de piano con RNN-RBM
Generador de música polifonica con RNN-RBM en TensorFlow, fork de [Music-Generation](https://github.com/cesern/Music_RNN_RBM).

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
