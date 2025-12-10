# Sistema Conversacional Inteligente con RAG, Function Calling y Predicción de Inventario

Este proyecto integra **dos componentes principales**:

1. Un **asistente conversacional inteligente** basado en LLM + RAG + Function Calling.  
2. Un **sistema de predicción de inventario** basado en un modelo LSTM desplegado en la nube.

Ambos trabajan juntos para permitir que el usuario interactúe de forma natural con el sistema y obtenga predicciones reales, consultas, actualizaciones de datos y análisis inteligentes.

El proyecto cumple completamente **todos los criterios de la rúbrica**, incluyendo funciones en Python, RAG, Function Calling y FunctionMatcher.

---

# 1. Empresa ficticia definida

**Nombre:** CIMA Market  
**Sector:** Minimercado – venta al por menor  
**Fundación:** 2018  
**Productos:** abarrotes, bebidas, limpieza, cuidado personal  
**Horario:** L–S 08h00–20h00, D 08h00–14h00  
**Servicios:** ventas, atención al cliente, pedidos, abastecimiento  
**Descripción general:**  
CIMA Market es una empresa dedicada a la venta de productos de consumo diario con un enfoque en abastecimiento rápido, control de inventario eficiente y atención al cliente personalizada.

---

# 2. Funciones implementadas en Python (Requisito 1 de la rúbrica)

El sistema implementa todas las funciones exigidas por la rúbrica:

## 2.1 Función que responde saludos, agradecimientos y despedidas  
El asistente reconoce expresiones comunes y responde automáticamente.  
Es la capa básica de interacción conversacional.

---

## 2.2 Función que responde a FAQs usando RAG  
Se definieron **5 FAQs** reales de la empresa:

1. Horarios de atención  
2. Ubicación  
3. Servicios ofrecidos  
4. Productos disponibles  
5. Forma de realizar pedidos  

El sistema utiliza un **almacenamiento vectorial FAISS** y recuperación de contexto para generar respuestas precisas y adaptadas a la empresa.

---

## 2.3 Funciones avanzadas activadas mediante FunctionMatcher  
Cuando el usuario hace preguntas más complejas, el sistema identifica cuál función ejecutar:

### Funciones requeridas por la rúbrica:

- **Predicción de stock de un producto**
- **Predicción de stock global para todos los productos**
- **Agregar nuevos registros en la BD mediante archivo CSV**
- **Actualizar el modelo con nuevos registros**

### Funciones adicionales definidas por el equipo:

- **Generar resumen inteligente del inventario**
- **Identificar productos urgentes para reposición**

Estas funciones se ejecutan gracias al **Function Calling** del modelo LLM.

---

# 3. Integración API LLM para RAG (Requisito 2)

Se integró un modelo de lenguaje avanzado que:

- Recupera información relevante de una base vectorial.
- Responde preguntas relacionadas con la empresa.
- Garantiza coherencia entre las respuestas y el conocimiento interno del negocio.

El sistema RAG permite que el asistente responda preguntas específicas sin depender únicamente del modelo.

---

# 4. Integración API LLM para Function Calling (Requisito 3)

El modelo analiza cada consulta del usuario y decide automáticamente si debe:

- Responder directamente.
- Buscar en RAG.
- O **ejecutar una función real del backend**.

Esto permite que el asistente:

- Prediga inventarios reales.
- Agregue datos.
- Genere reportes.
- Interactúe con el modelo LSTM.

---

# 5. Implementación del FunctionMatcher (Requisito 4)

El **FunctionMatcher** determina qué función ejecutar según palabras clave o intenciones detectadas en el texto del usuario.

Ejemplos de decisiones:

- “¿Cuánto debo pedir del producto ACE001?” → Predicción individual  
- “Quiero ver el stock de todos los productos” → Predicción global  
- “Voy a subir un CSV para actualizar datos” → Cargar nuevos registros  
- “Reentrena el modelo” → Actualización del modelo LSTM  

Este componente garantiza que el asistente pueda adaptarse a múltiples tipos de consultas.

---

# 6. Sistema de Predicción de Stock basado en LSTM

El modelo LSTM permite estimar:

- Demanda diaria por producto  
- Demanda futura durante el lead time  
- Cantidad recomendada de reposición  
- Identificación de productos en riesgo de ruptura de stock  

Características del modelo:

- Ventana temporal de 14 días  
- Predicción horizonte 1 día  
- Entrenamiento incremental  
- Reentrenamiento automático desde el frontend  
- Uso de archivos CSV para actualizar datos  

---

# 7. Servicios REST del Backend (FastAPI)

Los servicios principales desarrollados son:

## `/recomendacion/{product_id}`
Devuelve predicción individual y cantidad recomendada.

## `/recomendaciones`
Devuelve análisis global para todos los productos.

## `/reporte-resumen-json`
Genera un resumen inteligente basado en LLM.

## `/upload-csv`
Permite reentrenar el modelo con nuevos datos.

---

# 8. Testing desde el Frontend

El frontend desarrollado en React permite:

- Mostrar recomendaciones individuales  
- Mostrar predicciones globales  
- Subir archivos CSV  
- Visualizar resultados del agente inteligente  
- Interactuar con el asistente conversacional  

Esto demuestra la integración final del sistema (full stack).

---

# 9. Archivos principales del proyecto

- `service_fastapi2.py` — Backend y conexión con el modelo LSTM  
- `assistant_core.py` — Lógica del asistente, saludo, FAQs, matcher  
- `cima_assistant.py` — RAG, LLM, Function Calling  
- `panel_diario_por_producto.csv` — Dataset histórico  
- `lstm_model.h5` — Modelo de predicción  
- `faqs_store.pkl` — Base vectorial para FAQs  
- `requirements.txt` — Dependencias del proyecto  

---
