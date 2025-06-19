# Repositorio para los desafios de la materia de Procesamiento de Lenguaje Natural 

## Alumno: Sebastian Carreras

#### Los desafios se encuentran en las carpetas "Desafio/ejercicios" al final de cada enunciado.

##Consigna del desafío 1
1. Vectorizar documentos. Tomar 5 documentos al azar y medir similaridad con el resto de los documentos. Estudiar los 5 documentos más similares de cada uno analizar si tiene sentido la similaridad según el contenido del texto y la etiqueta de clasificación.

2. Entrenar modelos de clasificación Naïve Bayes para maximizar el desempeño de clasificación (f1-score macro) en el conjunto de datos de test. Considerar cambiar parámteros de instanciación del vectorizador y los modelos y probar modelos de Naïve Bayes Multinomial y ComplementNB.

3. Transponer la matriz documento-término. De esa manera se obtiene una matriz término-documento que puede ser interpretada como una colección de vectorización de palabras. Estudiar ahora similaridad entre palabras tomando 5 palabras y estudiando sus 5 más similares. La elección de palabras no debe ser al azar para evitar la aparición de términos poco interpretables, elegirlas "manualmente".

## Consigna del desafío 2

- Crear sus propios vectores con Gensim basado en lo visto en clase con otro dataset.
- Probar términos de interés y explicar similitudes en el espacio de embeddings.
- Intentar plantear y probar tests de analogías. Graficar los embeddings resultantes.
- Sacar conclusiones.

## Consigna del desafío 3
- Utilizar otro dataset y poner en práctica la generación de secuencias con las estrategias presentadas.

## Consigna del desafío 4
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
