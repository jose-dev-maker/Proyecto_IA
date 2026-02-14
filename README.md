Introducción
La diabetes es una enfermedad crónica que afecta a millones de personas en todo el mundo, y su detección temprana es crucial para prevenir complicaciones graves. El objetivo de este proyecto es desarrollar un sistema de Inteligencia Artificial capaz de predecir si un paciente tiene diabetes basándose en variables médicas diagnósticas.

Para ello, se ha utilizado un enfoque de aprendizaje supervisado, donde el algoritmo aprende de datos históricos etiquetados (pacientes con y sin diabetes) para clasificar nuevos casos. El conjunto de datos utilizado contiene variables como el nivel de glucosa, presión arterial, índice de masa corporal (BMI) y edad, entre otros.

Desarrollo
El proyecto se ha desarrollado en Python utilizando librerías estándar de ciencia de datos como Pandas, Scikit-Learn y Matplotlib. El flujo de trabajo ha seguido los siguientes pasos:

Preprocesamiento de Datos: Se detectó que en el dataset original, ciertas variables médicas (como glucosa o presión arterial) tenían valores "0", lo cual es biológicamente imposible y representa datos faltantes. Para solucionar esto, se reemplazaron los ceros por la mediana de cada columna respectiva. Posteriormente, los datos fueron normalizados mediante StandardScaler para asegurar que todas las variables tuvieran el mismo peso en el modelo.
Selección del Modelo: Se optó por implementar un clasificador Random Forest. Este algoritmo es una técnica de ensemble learning que combina múltiples árboles de decisión, lo que lo hace muy eficaz para reducir el sobreajuste (overfitting) y manejar relaciones no lineales entre los datos.
Entrenamiento y Evaluación: El dataset se dividió en un 80% para entrenamiento y un 20% para pruebas. El modelo fue entrenado con los datos de entrenamiento y se evaluó su rendimiento con los datos de prueba, utilizando métricas como la precisión (accuracy) y la matriz de confusión.
Conclusiones
El modelo desarrollado logró una precisión aceptable en la predicción de diabetes, demostrando que el aprendizaje automático puede ser una herramienta de apoyo valiosa en el diagnóstico médico.

Los resultados muestran que variables como la glucosa y el índice de masa corporal son determinantes en la predicción. El uso de técnicas de limpieza de datos fue fundamental; sin el reemplazo de los valores nulos (ceros), el modelo habría tenido un rendimiento significativamente peor. Como trabajo futuro, se podría mejorar el modelo balanceando las clases, ya que en el dataset hay más pacientes sin diabetes que con ella, lo que podría sesgar las predicciones hacia la clase mayoritaria.
