[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122265&assignment_repo_type=AssignmentRepo)
# Image Captioning
El objetivo de este proyecto es generar descripciones de imágenes con herramientas de deep learning.

Los [datos](https://www.kaggle.com/datasets/adityajn105/flickr8k) que se han utilziado para entrenar a la red neuronal consisten en 8.000 imagenes cada una pareada con 5 diferentes leyendas que proveen descripciones de las entidades y eventos de la imagen.

## Entorno de ejecución
Antes de ejecutar el código tienes que crear un entorno local con conda y activarlo. El archivo [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) tiene todas las dependencias necesarias. Ejecute el siguiente comando: ``conda env create -n NOMBREDEENTORNO --file environment.yml `` para crear un entorno conda con todas las dependencias requeridas y luego activarlo:
```shell
foo@bar:~$ conda env create -n ENVNAME --file environment.yml
foo@bar:~$ conda activate ENVNAME
```

To run the example code:
```shell
foo@bar:~$ python main.py
```
## Arquitectura
La arquitectura de nuestro modelo es la siguiente:

*  Un modelo Encoder, que dada una imagen la va codificando
*  Un modelo Atention, que extrae las características más relevantes de la imagen
*  Un modelo base de LSTM, utilizado para que mediante las características haga la generación de texto
*  Un modelo Decoder, que dada una imagen codificada, procede a descodificarla. En el proceso de descodificación, extrae las características mediante el modelo Atention y genera texto utilizando el LSTM


## Dataset preprocessing
Tras haber separado en directorios las imagenes para realizar un train_test_split. Les aplicamos:
Transformaciones de resizing, random crop y normalización.
Tenemos pensado aplicar data augmentation

## Train
Hasta el momento se han probado ejecuciones residuales, de comprobación de ejecución con diferentes metodos (para poder reconstruir maneras de evaluar el modelo, diferentes metricas, ...)

## Results
Results

## Contributors
Write here the name and UAB mail of the group members

- Eric Alcaraz del Pico- 1603504@uab.cat
- Raül Yusef Dalgamoni Alonso - 1599225@uab.cat
- Roger Vera Filella - 1600785@uab.cat

Xarxes Neuronals i Aprenentatge Profund

Grau de Data Engineering,

UAB, 2023
