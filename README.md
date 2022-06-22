# Graph-Based Siamese Network

Código y documentación para el modelo Graph-Based Siamese Network, descrito en el artículo:

[Graph-Based Siamese Network for Authorship Verification](http://ceur-ws.org/Vol-2936/paper-165.pdf)

En la carpeta `slides` puedes encontrar presentaciones del modelo propuesto. Es un buen recurso para comenzar a entender la tarea y la idea general.

# Requisitos

Este código fue desarrollado en python 3.8. El repositorio contiene un archivo dependency.txt, donde se explica las dependencias necesarias, donde descargarlas y como instalarlas.

Nota.
Dado que es un proyecto desarrollado utilizando Pytorch CUDA es necesario contar con una GPU Nvidia, es decir, no se podrá instalar Pytorch CUDA con una tarjeta AMD, como las que utiliza MAC. Para solucionar esto se podría instalar Pytorch solo con CPU. 

No olvides activar el ambiente con:
```
conda activate <env_name>
```

# Prueba

Para realizar una prueba de que todo se instaló correctamente, posiciónate en la carpeta principal del repositorio
y ejecuta el comando

```
python codes/main.py --cfg 'codes/cfg/GBSN_test.yml'
```

Este comando ejecuta el script `main.py`, que se encuentra dentro de la carpeta `codes`. Dicho script necesita un archivo de configuración,
indicado con el parámetro `--cfg`. El archivo de configuración es un archivo en formato yml; en este caso, se encuentra en la carpeta
codes/cfg y tiene nombre `GBSN_test.yml`.

# Archivo de configuración

El archivo de configuración se espera en formato yml. Este formato permite indicar
estructuras del tipo diccionario (llave, valor) y listas de una forma visualmente más sencilla que un formato json.

Este es un ejemplo de un archivo de configuración:

```
CONFIG_NAME: 'GBSN_test'

# ========== General options
device: 'cuda:0'
dataset_name: '22-train'
folder_sufix: '_None_1046'
doc_dict_folder_prefix: '../data/PAN22_graphs/'
ds_list_folder_prefix: '../data/PAN22_text_split/'
dest_folder_prefix: '../test/test_output_siamese/test_command_line/'
epochs: 100
lim:
checkpoint_freq: 100
bm_free_epochs: 1
lr: 0.001
batch_size: 256
num_workers: 4

# ========== Datasets options
ds_op:
    exp_label: 'short'

# ========== Model options
exp_ops:
  #- model_class: SiameseNetwork
  - model_args:
        raw_components_list:
          - class: 'GBFeatures'
            args:
                conv_layers_num: 2
                conv_type: 'gnn.LEConv'
                h_ch: 64
                out_ch: 64
                pool_type: 'GlobalAttentionSelect'
                pool_att_ch: 32
                pool_att_layers: 2
                pool_ref: 'last'
        final_out_join: 'abs'
        final_out_layers_num: 2
        final_out_ch: 64
repeat_experiment: 1

# ========== Loss options
main_loss: 'soft_f1_loss'
```

Parámetros:

* device: indica la GPU utilizada para ejecutar el modelo, comando para listar las GPUs disponibles:
```
gpustats
```
* dataset_name: nombrel dataset que se encuentre dentro de la carpeta data/PAN22_text_split/
* folder_sufix: sufijo para el directorio que se encuentra en data/PAN22_graphs/
* doc_dict_folder_prefix: directorio que contiene los textos como grafos
* ds_list_folder_prefix: directorio que contiene las particiones, textos, autores etc.
* dest_folder_prefix: directorio en donde se guaradán los resultados.
* epochs: epocas que el modelo ejecutará.
* lim: parámetro de debug que corta el dataset a un rango, solo para utilizar en pruebas en caso de que el equipo no tenga muchos recursos, para tomar todo el dataset dejar vacío.
* lr: tasa de aprendizaje
* exp_label: tipo de gráfica a utilizar (short,med,full)
* conv_layers_num: numero de capas convolucionales, para la etapa de node embedding.
* conv_type: tipo de capa convolucional (gnn.LEConv,gnn.GraphConv,gnn.GCN2Conv,gnn.TAGConv).
* pool_att_layers: Numero de capas para la etapa de pooling.
* final_out_layers_num: Numero de capas para el clasificador
* main_loss: función de pérdida a utilizar (ver archivo siamese_graph_trainer.py linea 189).

# Outputs

Los resultados de esta ejecución los podemos encontrar en la carpeta
`test/test_output_siamese/test_command_line/`. Dicha carpeta puede cambiarse en
el archivo de configuración.

Los archivos generados son:
* Logs (extensión `.txt`): El archivo tiene en su nombre las características
  del experimento `GBSN[fecha][hora]_[gráfica_usada]_[repetición]`. Resume los
  parámetros utilizados, la arquitectura del modelo, los tiempos de ejecución,
  la pérdida y métricas obtenidas por los mejores y el último modelo y las
  métricas después del ajuste de threshold.
* Modelos (extensión `.pth`): Estos archivos son checkpoints del estado del
  modelo en cierta época (`checkpoint_ ...`), el estado del modelo que obtuvo
  menor pérdida en validación (`best_model_`) o bien el modelo que obtuvo mejor
  promédio en las métricas (`best_model_sa_`).
* Métricas (extensión `.pkl`): Son archivos con las métricas obtenidas por
  nuestros modelos. Todos los archivos comienzan con el nombre del modelo y los
  distingue un sufijo: `_metrics` es un dataframe con las métricas obtenidas
  por el modelo en entrenamiento, validación y prueba antes de ningún ajuste de
  threshold, `threshold-best_model ...` son las predicciónes y métricas que se
  calculan para realizar el ajuste de threshold.
* Imágenes (extensión `.png`): Son visualizaciones de los datos obtenidos,
  todas comienzan con el nombre del modelo y las distingue el sufijo utilizado.
  `_loss` y `_metrics` grafican las pérdidas y el promedio de las métricas
  respecto las épocas, respectivamente. `_best_model_th-val_[métrica]` grafica
  los puntajes calculados para el ajuste de threshold respecto una de las
  métricas para validación; similarmente `_best_model_th-test_[métrica]` lo
  grafica para el conjunto de prueba. `_best_model_pred_test_raw` y
  `_best_model_pred_test` muestran histogramas de la distribución de las
  respuestas del modelo antes y después del ajuste de threshold en el conjunto
  de prueba.

# Para desarrolladores

La estructura de este proyecto es la siguiente:
* `codes`: Contiene los scripts y archivos de configuración para ejecutar el
  modelo. Más adelante describiremos algunos más a detalle
* `test`: Contiene scripts para probar el buen funcionamiento del proyecto.
* `data`: Contiene los datos sobre los cuales se entrenan los modelos. En el
  repositorio solamente se incluyen un dataset con 1000 parejas de textos para
  ejecutar pruebas.
* `PAN22_predict`: Contiene los mejores modelos presentados en el PAN y
  archivos que se utilizaron para probar que nuestra entrega funcionara bien.
* `statistic_analisys`: Contiene visualizaciones de los puntajes obtenidos a
  lo largo de todos nuestros experimentos.

Respecto a las particiones utilizadas para entrenar, validar y probar el modelo, se pueden 
generar con el siguiente comando.

```
python codes/datasets_PAN.py
```

En el script datasets_PAN, se encuentran todas los métodos necesarios para transformar 
todos los textos a un diccionario, obtener estadísticos del dataset y las particiones:

* ds_list_train_n.pkl
* ds_list_val_n.pkl
* ds_list_test_n.pkl

Que corresponden a las particiones de entrenamiento, validacion y prueba respectivamente, los
cuales no comparten autores. Se generará un directorio nuevo en `data/PAN22_text_split` en donde se encontrará un directorio con estadísticos y particiones.

Ya que el modelo es basado en grafos, es necesario representar los textos como grafos, para mayor información se puede consultar la carpeta `slides`, donde se detalla como se representan los 
textos como grafos.
Para generar estas representaciones es necesario ejecutar e siguiente comando.

```
python codes/dataset_to_graph.py
```
Se generará un directorio nuevo en `data/PAN22_graphs` en donde se encontrará un directorio con los
archivos pkl con las representaciones de los textos.


# Cita

Embarcadero-Ruiz, D., Gómez-Adorno, H., Embarcadero-Ruiz, A., & Sierra, G. (2022). Graph-based Siamese network for authorship verification. Mathematics, 10(2), 277.


