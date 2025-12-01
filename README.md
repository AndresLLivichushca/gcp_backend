# Sistema Inteligente de Predicción de Stock con LSTM, FastAPI y LangChain

Este proyecto implementa un sistema completo para la predicción de demanda, recomendación de stock, reentrenamiento automático y análisis inteligente del inventario mediante un agente de lenguaje natural.  
El backend está desplegado en una máquina virtual en Google Cloud y expone servicios REST consumidos por un Frontend.

---

## Descripción General

El sistema utiliza un modelo LSTM entrenado sobre series temporales de ventas para estimar la demanda diaria de productos y recomendar cantidades óptimas de compra.

Incluye las siguientes capacidades:

- Modelo LSTM desarrollado con TensorFlow/Keras.
- Backend REST desarrollado con FastAPI.
- Reentrenamiento automático mediante carga de CSV.
- Agente inteligente desarrollado con LangChain y OpenAI.
- Parsing estructurado mediante Pydantic.
- Ejecución y despliegue en Google Cloud.

---

## Tecnologías Utilizadas

- Python 3.10  
- FastAPI  
- TensorFlow / Keras  
- Pandas y NumPy  
- LangChain  
- OpenAI API  
- PydanticOutputParser  
- Uvicorn  
- Google Cloud VM  

---

## Servicios REST Disponibles

### 1. `/recomendacion/{product_id}`
Devuelve la predicción diaria y la cantidad recomendada de compra para un producto específico.  
Acepta parámetros como fecha, lead time y nivel de servicio.

---

### 2. `/recomendaciones`
Genera recomendaciones de compra para todos los productos con inventario insuficiente según el modelo LSTM.

---

### 3. `/reporte-resumen-json`
Genera un resumen inteligente en lenguaje natural que incluye:

- Productos urgentes.
- Cantidades recomendadas de compra.
- Conclusión analítica generada por el agente LangChain.

Este servicio utiliza un modelo de lenguaje configurado con:

- ChatOpenAI  
- PromptTemplate  
- PydanticOutputParser  

Lo que garantiza una respuesta en formato estructurado y un texto claro en español sencillo.

---

### 4. `/upload-csv`
Servicio que permite cargar un archivo CSV para reentrenar el modelo.  
Acciones que realiza:

1. Validación del archivo.
2. Normalización de fechas.
3. Unión con el dataset existente.
4. Ordenamiento por fecha y producto.
5. Construcción de secuencias para entrenamiento.
6. Reentrenamiento del modelo LSTM.
7. Guardado del modelo actualizado.
8. Recarga del modelo dentro del backend.

Permite mantener actualizado el modelo sin interrumpir el servicio.

---

## Funcionamiento del Modelo LSTM

El modelo está entrenado para predecir la demanda diaria por producto.  
Características del modelo:

- Ventana temporal (lookback) de 14 días.
- Predicción a horizonte 1 día.
- Entrenamiento incremental al recibir un nuevo CSV.
- Features utilizadas definidas en `features_used.json`.

El modelo genera:

- Predicción de ventas diarias.
- Demanda durante el lead time.
- Stock objetivo.
- Cantidad recomendada de compra.

---

## Agente Inteligente con LangChain

El agente de análisis utiliza:

- ChatOpenAI (modelo gpt-4o-mini).
- PromptTemplate para guiar la respuesta.
- PydanticOutputParser para garantizar una estructura específica.

El agente produce un resumen en lenguaje natural, comprensible para cualquier usuario sin conocimientos técnicos en inventario o machine learning.

---

## Reentrenamiento Automático

El sistema permite actualizar el modelo mediante la carga de un archivo CSV.  
El proceso incluye:

- Validación estricta de columnas.
- Conversión y validación de fechas.
- Combinación con el panel existente.
- Reconstrucción completa de secuencias.
- Entrenamiento incremental del modelo.
- Actualización de `lstm_model.h5`.

---

## Despliegue en Google Cloud

El backend se ejecuta en una VM con:

- Ubuntu  
- Python 3.10  
- Uvicorn + FastAPI  

El despliegue se actualiza mediante Git:

git pull origin main
pkill uvicorn
uvicorn service_fastapi2:app --host 0.0.0.0 --port 8000 --reload


El servicio queda accesible desde la IP pública de la máquina virtual.

---

## Testing desde el Frontend

El Frontend utiliza los endpoints para:

- Seleccionar fecha del análisis.
- Mostrar inventario urgente.
- Visualizar cantidades recomendadas.
- Mostrar el resumen generado por el agente LangChain.
- Validar cambios después del reentrenamiento.

Esto cumple el criterio de pruebas funcionales exigido en la rúbrica.

---

## Cumplimiento de la Rúbrica

Este proyecto cumple completamente los cuatro criterios exigidos:

1. Optimización y documentación del método.  
2. Desarrollo y despliegue de servicios de predicción en la nube.  
3. Implementación de servicio para carga de datos y reentrenamiento.  
4. Testing desde el Frontend.  

El proyecto se encuentra en el nivel máximo de desempeño.

---

## Archivos Principales del Proyecto

| Archivo                        | Descripción                                        |
|-------------------------------|----------------------------------------------------|
| `service_fastapi2.py`         | Backend principal con FastAPI y LangChain          |
| `panel_diario_por_producto.csv` | Dataset histórico                                 |
| `lstm_model.h5`               | Modelo LSTM entrenado y actualizado                |
| `features_used.json`          | Columnas utilizadas como features                  |
| `data/`                       | Carpeta con datos                                   |
| `models/`                     | Carpeta con archivos de modelo                     |

---

## Licencia

MIT License.
