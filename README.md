[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122265&assignment_repo_type=AssignmentRepo)
# Image Captioning
El objetivo de este proyecto es desarrollar un modelo Deep Learning que sea capaz de generar a partir de una imagen un texto que describa que esta pasando en esa imagen.

## Dataset

Los [datos](https://www.kaggle.com/datasets/adityajn105/flickr8k) que se han utilziado para desarrollar el proyecto provienen de la página web Kaggle. Dichos datos contienen alrededor de 8000 imágenes. Este dataset, a parte de las imágenes incluye «captions». En un archivo txt, se incluye por cada imágen entre 3 y 5 captions, que describen de manera sencilla que es lo que está sucediendo en la imágen. A continuación podemos ver algunos ejemplos:

![Imagen Ejemplo con diferentes captions](src/Plot_Inicial-Corrected.jpg)

Una vez descargados los datos mediante la interfaz de Kaggle, se ha procedido a realizar un split para tener un conjunto de train y otro de validación. De este modo, se ha realizado una separación 80/20, la separación se ha hecho de las caption, por lo que la estructura del dataset es la siguiente:

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

En primer lugar, el funcionamiento del Encoder es el siguiente:

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
En esta sección se hablará sobre todo el proceso realizado en el train, se explicará el proceso de entrenamiento, la arquitectura de este, validación y pruebas, así como los hiperparámetros utilizados durante el entrenamiento. Además, se explicará la funció de pérdida, optimización y métricas empleadas.

### Proceso de entrenamiento
Aqui explicar un par de cosa:
1. como se entrena
   1. Aqui podemos poner que usamos el 80% de las imagens para entrenar y que cuando acabamos 1 epoca, con el 20% restante hacemos una validacion para ver como esta aprendiendo el modelo
2. que arquitectura seguimos para entrenar
   1. una vez explicado el como funciona el proceso de entrenamiento, pues podemos adentrarnos a poner una pincellada de la arquitectura aunque se haya explicado anteriormente, pero no hace falta definir encoder ni decoder ya que se ha explicado.


### Función de pérdida y optimización
La función de loss utilizada es la de CrossEntropy, se intentó probar la función de pérdida de divergencia KL (Kullback-Leibler), pero esta función de pérdida no tiene la propiedad de gradiente eficiente, esto significa que durante el entrenamiento esta propiedad facilita la propagación del error a través de las capas de la red neuronal, por lo que dificulta que se pueda optimizar. Por lo tanto, finalmente se decidió usar la Crossentropy, ya que la Crossentropy es la más idonea para este tipo de problemas, debido a que se puede considerar cada palabra como una etiqueta de clasificación, eso permite que el modelo pueda predecir la siguiente palabra en función de la imagen de entrada y las palabras anteriores. 

La función de optimización inicial era Adam, Adam ya daba un resultado bueno, pero aún así se quiso probar diversos optimizadores para comprobar si era el mejor, o habia alguno que se adaptara mejor. Primero de todo se usó el optimizador SGD, debido a que se queria probar el hyperparámtero Cyclic en Learning rate, y para ello era necesario un optimizador con momentum,pero el resultado de este no fue mejor que el de Adam y se descartó. Además de probar el SGD, se probó Adagrad y Adadelta, pero ninguno de los mejoró la ejecución inicial.


### Metricas de evaluación
Para el proyecto se emplearon varias métricas para medir el rendimiento del modelo de image captioning. Algunas de las métricas utilizadas son: BLEU, Perplexity, ROUGE y coeficiente Jaccbard.
De estas cuatro métricas mencionadas anteriormente, solo se hizo uso de dos en el train y una adicional en el test, la otra se descartó. Para el train se ueó la BLEU y Perplexity:
   - BLEU: Se empleó la métrica BLEU de Pytorch en el problema, para evaluar la similitud entre las captions predichas por el modelo y las captions reales que proporciona el dataset. Esta métrica, compara los n-gramas que están en los caption predichos con los reales. La fórmula para el cálculo de la BLEU és la siguiente: [PONER FOTO DE LA FORMULA DE BLEU]
   - Perplexity: Se empleó la métrica Perplexity para evaluar la calidad de las captions predichas. Esta métrica se calcula mediante el cáculo de la probabilidad de las captions predichas del modelo, y mide que tan bien el modelo puede llegar a predecir las siguiente palabras. La fórmula como tal, es la siguiente: [PONER FOTO DE LA FORMULA DE PERPLEXITY]

Tal i como se ha mencionado anteriormente, se hizo uso de una métrica adicional en el test, esta métrica es el coefficiente de Jaccbard:
   - Coeficiente de Jaccbard: El coeficiente de jaccbar calcula la similitud entre dos conjuntos, dividiendo la longitud de la intersección por la unión de estos.

{AQUI TENEMOS QUE PONER QUE TODAS LAS METRICAS NOS HAN DADO MAL??? O ESTO VA EN RESULTADOS??}

### Hyperparámetros

Teniendo en cuenta que los hyperparametros base han sido los siguientes:

- Encoder: Resnet50
- Encoder dimension: 2048
- Attention dimension: 256
- Embedding size: 300
- LSTM dimension: 512
- Learning rate: 0.0003
- Optimizer: Adam
- Loss: CrossEntropyLoss


Adicionalmente se han probado diferentes configuraciones:
- Resnet50/101
- Embbedding size
- Attention dimension
- LSTM dimension
- Learning rate: 0.1, 0.01, CyclicLR, LambdaLR

El código, inicialmente venia con la resnet50, pero se quiso probar el modelo resnet101, ya que se pensaba que al haber más capas, el modelo ajustaria mejor los pesos iniciales y de esta forma mejoraría la predicción, pero resultó que la mejoria no era tan significante como para canviar el modelo, y además el tiempo de ejecución augmentaba significativamente respecto al de resnet50, estas dos cosas llevaron a la reflexión de que el modelo resnet50 era suficientemente bueno para el proyecto.

Embbedding size es un parámetro que se usa para representar palabras o letras en un espacio vectorial. Al augmentarlo puede mejorar la representació debido a que permite capturar mas información en los vectores. Y eso ocurrió, se agumentó el embbeding size de 300 a 1024, cosa que propició una mejora en el modelo, debido a que al augmentarlo se capturaron mas detalles y por tanto llevo a una comprensión mas detallada del lenguaje. 

Cuando se habla de attention size se refiere al tamaño de los vectores de atención utilizados. Se utiliza para asignar pesos a diferentes partes de una secuencia de entrada, en este caso, el output del encoder de la imagen. Para ver si el resultado se podía mejorar, se augmentó este, el attention size tiene unas características muy similares al embedding size, por la qual cosa al agumentarlo ocurre lo mismo que con el embbedding size, es capaz de captar mas detalle de lo que le entra.

El LSTM size es el mismo que el decoder size, y este se comporta de la misma manera que los dos anteriores, pero al augmentalo puede tener una complejidad mayor y requerir mas recursos de cómputo, por la qual cosa no mejoraba significativamente el modelo y se obtó por no modificar el tamaño.


El learning rate ha sido el hyperparámetro que mas se ha modificado para ver si habia mejoria o no, se disminuyó, se augmentó e incluso se hizo un learning rate schedule. Con todas las prubas realizadas, se llegó a una conclusión, que el mejor learning rate para el modelo, era el que venia por defector, debido que al augmentarlo overfitteaba, al disminuirlo, no mejoraba al que ya teníamos y al hacer el schedule, se intentaba adaptar al maximo a los datos, que al final no aprendia nada y underfitteaba.

Por tanto, con todo lo que se ha explicado anteriormente, los hyperparámetros escojidos para hacer el mejor modelo posible han sido:

- Encoder: Resnet50
- Encoder dimension: 2048
- Attention dimension: 1024
- Embedding size: 1024
- LSTM dimension: 512
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Learning rate: 0.0003

### Tiempo de entrenamiento y recursos
Aqui yo pondria que dura cada epoca, que recursos usamos, podriamos poner alguna fotito de Wanb ya q estos tienen graficas viendo el rendimiento del problema, pero sino este punto se puede eliminar





## Results
Aqui podemos poner los resultados de las losses y metricas del modelo que consideramos mejor, algunas predicciones con el validation y por ultimo explicar que hemos utlizado unas fotos nuestras para predecir.

Podemos poner possibles mejoras si quereis

## Referencias
Poner algunas referencias 


## Contributors

- Eric Alcaraz del Pico- 1603504@uab.cat
- Raül Yusef Dalgamoni Alonso - 1599225@uab.cat
- Roger Vera Filella - 1600785@uab.cat

Xarxes Neuronals i Aprenentatge Profund

Grau de Data Engineering,

UAB, 2023
