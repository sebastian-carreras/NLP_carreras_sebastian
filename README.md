# Desafíos de la materia de Procesamiento de Lenguaje Natural

## Desarrolladores
- Sebastian Carreras

#### Los desafios se encuentran en las carpetas "Desafio/ejercicios" al final de cada enunciado.

---

## Desafío 1
- Vectorizar documentos. Tomar 5 documentos al azar y medir similaridad con el resto de los documentos. Estudiar los 5 documentos más similares de cada uno analizar si tiene sentido la similaridad según el contenido del texto y la etiqueta de clasificación.
- Entrenar modelos de clasificación Naïve Bayes para maximizar el desempeño de clasificación (f1-score macro) en el conjunto de datos de test. Considerar cambiar parámteros de instanciación del vectorizador y los modelos y probar modelos de Naïve Bayes Multinomial y ComplementNB.
- Transponer la matriz documento-término. De esa manera se obtiene una matriz término-documento que puede ser interpretada como una colección de vectorización de palabras. Estudiar ahora similaridad entre palabras tomando 5 palabras y estudiando sus 5 más similares. La elección de palabras no debe ser al azar para evitar la aparición de términos poco interpretables, elegirlas "manualmente".

### Temas: 
Clasificación de texto, vectorización de características con TF-IDF (Term Frequency-Inverse Document Frequency), uso de N-gramas para la extracción de características, optimización de hiperparámetros mediante GridSearch, y construcción de flujos de trabajo de machine learning con Pipelines de Scikit-learn.

### Modelos desarrollados: 
Naive Bayes Multinomial (MultinomialNB) y Complement Naive Bayes (ComplementNB).

### Notebook: 
- https://github.com/chatoxz-wp/NPL_carreras_sebastian/blob/main/Desafio%201/ejercicios/Desafio_1.ipynb

---

## Desafío 2
- Crear sus propios vectores con Gensim basado en lo visto en clase con otro dataset.
- Probar términos de interés y explicar similitudes en el espacio de embeddings.
- Intentar plantear y probar tests de analogías. Graficar los embeddings resultantes.
- Sacar conclusiones.

### Temas: 
Procesamiento de lenguaje natural (NLP), creación de embeddings de palabras (word embeddings) a partir de un corpus textual, búsqueda de similitud semántica entre palabras, y desarrollo de un bot conversacional simple (chatbot) basado en la proximidad vectorial.

### Modelos desarrollados: 
Modelo Word2Vec, implementado con la librería Gensim para aprender las representaciones vectoriales de las palabras

---

## Desafío 3
- Seleccionar un corpus de texto sobre el cual entrenar el modelo de lenguaje.
- Realizar el pre-procesamiento adecuado para tokenizar el corpus, estructurar el dataset y separar entre datos de entrenamiento y validación.
- Proponer arquitecturas de redes neuronales basadas en unidades recurrentes para implementar un modelo de lenguaje.
- Con el o los modelos que consideren adecuados, generar nuevas secuencias a partir de secuencias de contexto con las estrategias de greedy search y beam search determístico y estocástico. En este último caso observar el efecto de la temperatura en la generación de secuencias.

### Temas: 
Modelado de lenguaje, generación de texto, redes neuronales recurrentes (RNN), preprocesamiento y tokenización de un corpus para modelos neuronales, y estrategias de decodificación de secuencias como Greedy Search, Beam Search (determinístico y estocástico), y el ajuste de la aleatoriedad mediante el parámetro de temperatura.

### Modelos desarrollados: 
Red Neuronal Recurrente con una arquitectura secuencial que incluye capas de Embedding, LSTM (Long Short-Term Memory) y una capa Densa con activación Softmax para la predicción de la siguiente palabra

### Sugerencias
- Durante el entrenamiento, guiarse por el descenso de la perplejidad en los datos de validación para finalizar el entrenamiento. Para ello se provee un callback.
- Explorar utilizar SimpleRNN (celda de Elman), LSTM y GRU.
- rmsprop es el optimizador recomendado para la buena convergencia. No obstante se pueden explorar otros.

---

## Desafío 4
- Implementar un modelo de traducción automática neuronal (Neural Machine Translation) utilizando la arquitectura Encoder-Decoder con LSTM.
- Construir QA Bot basado en el ejemplo del traductor pero con un dataset QA LINK
### Recomendaciones:
- MAX_VOCAB_SIZE = 8000
- max_length ~ 10
- Embeddings 300 Fasttext
- n_units = 128
- LSTM Dropout 0.2
- Epochs 30~50
### Preguntas interesantes:
- Do you read?
- Do you have any pet?
- Where are you from?

### Temas: 
Traducción automática neuronal (Neural Machine Translation), modelos Sequence-to-Sequence (Seq2Seq), la arquitectura Encoder-Decoder, el uso de redes LSTM para codificar la secuencia de entrada y decodificarla en el idioma de salida, y la implementación de modelos separados para el entrenamiento y la inferencia.

### Modelos desarrollados:
Modelo Encoder-Decoder con LSTMs implementado con la API funcional de Keras. Se construyen tres modelos distintos: un modelo completo para el entrenamiento, un modelo de solo Encoder y un modelo de solo Decoder para realizar las traducciones en la fase de inferencia