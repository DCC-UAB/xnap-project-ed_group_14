[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122265&assignment_repo_type=AssignmentRepo)
# Image Captioning
El objetivo de este proyecto es desarrollar un modelo Deep Learning que sea capaz de generar a partir de una imagen un texto que describa que esta pasando en esa imagen.

## Dataset

Los [datos](https://www.kaggle.com/datasets/adityajn105/flickr8k) que se han utilziado para desarrollar el proyecto provienen de la página web Kaggle. Dichos datos contienen alrededor de 8000 imágenes. Este dataset, a parte de las imágenes incluye «captions». En un archivo txt, se incluye por cada imágen entre 3 y 5 captions, que describen de manera sencilla que es lo que está sucediendo en la imágen. A continuación podemos ver algunos ejemplos:

![Imagen Ejemplo con diferentes captions](src/Plot_Inicial-Corrected.jpg)

Una vez descargados los datos mediante la interfaz de Kaggle, se ha procedido a realizar un split para tener un conjunto de train y otro de validación. De este modo, se ha realizado una separación 80/20, por lo que la estructura del dataset es la siguiente:

|               | Imágenes  | Captions  |
| ---------     | --------- | --------- |
| Train         | 6473      | 32364     |
| Validación    |  1619     | 8091      |
| Total         | 8091      | 40455     |


## Dataset preprocessing
Tras haber separado en directorios las imagenes para realizar un train_test_split. Les aplicamos:
Transformaciones de resizing, random crop y normalización. En primer lugar aplicamos un resizing, disminuyendo el tamaño de la imagen a 226. Después realizamos un random crop a 224. Una vez redimensionada la imagen la convertimos a tensor (convritiendo los valores de 0 a 1). Finalmente, aplicamos una normalización con valores predeterminados dados por el modelo base. 
Una vez se han aplicado estas transformaciones a cada uno de las imágenes del dataloader, ya están aptas para ser transferidas al modelo.

## Entorno de ejecución
Antes de ejecutar el código tienes que crear un entorno local con conda y activarlo. El archivo [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) tiene todas las dependencias necesarias. Ejecute el siguiente comando: ``conda env create -n NOMBREDEENTORNO --file environment.yml `` para crear un entorno conda con todas las dependencias requeridas y luego activarlo:
```shell
conda env create -n ENVNAME --file environment.yml
conda activate ENVNAME
```

To run the example code:
```shell
python main.py
```
## Arquitectura
La arquitectura de nuestro modelo es la siguiente:

*  Un modelo Encoder, que dada una imagen la va codificando
*  Un modelo Atention, que extrae las características más relevantes de la imagen
*  Un modelo base de LSTM, utilizado para que mediante las características haga la generación de texto
*  Un modelo Decoder, que dada una imagen codificada, procede a descodificarla. En el proceso de descodificación, extrae las características mediante el modelo Atention y genera texto utilizando el LSTM

![Imagen Arquitectura del modelo](src/Model.png)

En primer lugar, el funcionamiento del Encoder es el sigueinte:

El modelo EncoderCNN se basa en la arquitectura de red neuronal convolucional ResNet-101 pre-entrenada. El propósito de este modelo es extraer características visuales significativas de las imágenes de entrada. 

La arquitectura ResNet-101 consta de múltiples capas convolucionales y de agrupación que permiten aprender representaciones jerárquicas de las imágenes. Estas capas convolucionales son altamente efectivas para capturar características visuales, como bordes, texturas y formas. Al cargarla, se ha congelado el aprendizaje y se han eliminado als 2 últimas capas. También se ha definido una capa lineal. Finalmente, se retorna el tensor de características resultante, que representa una representación visual compacta de la imagen de entrada.

Otro modelo implementado es el Attnention, cuyo funcionamineto es el siguiente:

El modelo Attention se utiliza para calcular pesos de atención sobre las características extraídas por el modelo EncoderCNN y el estado oculto del modelo Decoder. La atención se utiliza para enfocarse en regiones específicas de las características visuales durante el proceso de generación de subtítulos.

En primer lugar, hay 3 capas, W, U, A que reciben de dimensión de entrada la dimensión del decoder, dimensión del encoder y la dimesnión del attention, respectivamente. Y producen una salida de dimensión del attention en los dos primeros casos y un único valor en el último.
Las salidas de las dos primeras capas son enviadas a la terceera (A), tras haber aplicado una tangente hiperbólica con tal de unirlas. Tras la salida de la última capa se aplica una Softmax.

![Heatmap de attentions](src/Attention.png)

Por último se tiene un decoder, el cual tiene este funcionamiento:

El modelo DecoderRNN es responsable de generar captions basados en las características visuales y las representaciones de atención generadas por el modelo EncoderCNN y el modelo Attention, respectivamente. En primer lugar se crea una capa embedding para mapear las palabras del bocabulario a vectores. También se definen capas lineales para inicalizar el hidden state del al LSTM, que hace un procesamiento recurrente. Una capa fully connected genera las predicciones, y finalmente se aplica una capa de dropout.

![Imagen Predicha](src/Prediction.png)


## Train
Hasta el momento se han probado ejecuciones residuales, de comprobación de ejecución con diferentes metodos (para poder reconstruir maneras de evaluar el modelo, diferentes metricas, ...)

Teniendo en cuenta que los hyperparametros base han sido los siguientes:

- Encoder: Resnet50
- Encoder dimension: 2048
- Attention dimension: 256
- Decoder dimension: 512
- Embedding size: 300
- LSTM dimension: 512
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Learning rate: 0.0003

Adicionalmente se han probado diferentes configuraciones:
- Resnet50/101
- LSTM dimension
- Embbedding size
- Optimizers: SGD, ADAGRAD, ADADELTA
- Learning rate: 0.1, 0.01, CyclicLR, LambdaLR

## Results
Results


## Contributors

- Eric Alcaraz del Pico- 1603504@uab.cat
- Raül Yusef Dalgamoni Alonso - 1599225@uab.cat
- Roger Vera Filella - 1600785@uab.cat

Xarxes Neuronals i Aprenentatge Profund

Grau de Data Engineering,

UAB, 2023
