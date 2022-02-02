# AuthorshipVerification-GraphDeepLearning

Este repositorio contiene las implementaciones de los baselines propuestos para la tarea de verificación de autoría del [PAN2021](https://pan.webis.de/clef21/pan21-web/author-identification.html).

El método basado en compresión de textos (compression method calculating cross-entropy), se propone en [Teahan & Harper](https://link.springer.com/chapter/10.1007/978-94-017-0171-6_7). La implementación contenida en este repositorio es una ligera modificación: en lugar de proponer una regla de decisión basada en un umbral, se usa una regresión logística para hacer la clasificación.



## BOWSVM
***

En esta carpeta se encuentra el código necesario para correr un SVC con parámetros predeterminados para realizar una comparación con otros modelos para la tarea de verificación de autoría del PAN2021.

## CosineSimilarity
***

En esta carpeta se encuentra una solución rápida a la tarea PAN2020 sobre verificación de autoría. Todos los documentos se representan usando un modelo de Bag of character ngrams, eso es TFIDF ponderado. La semejanza del coseno entre cada par de documentos en el conjunto de datos de calibración es calculado. Finalmente, las similitudes resultantes son optimizadas y proyectadas a través de un simple reescalado, para que puedan funcionar como pseudo-probabilidades, que indican la probabilidad de que un par de documentos es un par del mismo autor.