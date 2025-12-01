# ============================================
# 1. IMPORTS Y CONFIGURACIÓN
# ============================================
import os
import re
import json
from typing import List, Dict, Any, Optional

import requests
from openai import OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Cliente OpenAI (usa la variable de entorno OPENAI_API_KEY)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# URL del backend FastAPI de stock
BACKEND_URL = "http://127.0.0.1:8000"   # cambia si usas otra IP

# ============================================
# 2. NICHO DE EMPRESA Y DATOS BÁSICOS
# ============================================

COMPANY_INFO = {
    "nombre": "CIMA Market",
    "anio_fundacion": 2015,
    "hitos": [
        "Apertura de la primera tienda en el barrio central",
        "Implementación de sistema de inventario automatizado",
        "Lanzamiento de servicio de pedidos por WhatsApp",
    ],
    "productos": [
        "Abarrotes (arroz, azúcar, fideos, aceite)",
        "Bebidas (agua, gaseosas, jugos)",
        "Limpieza (detergentes, jabones, desinfectantes)",
    ],
    "servicios": [
        "Atención en tienda",
        "Pedidos a domicilio en el barrio",
    ],
    "horario_atencion": "Lunes a domingo, de 7h00 a 21h00",
}


def get_company_info() -> str:
    """Devuelve una descripción general de la empresa ficticia."""
    return (
        f"{COMPANY_INFO['nombre']} es un minimercado de barrio fundado en "
        f"{COMPANY_INFO['anio_fundacion']}. Ofrecemos productos como "
        f"{', '.join(COMPANY_INFO['productos'])}. Nuestro horario de atención es "
        f"{COMPANY_INFO['horario_atencion']}."
    )

# ============================================
# 3. SALUDOS / AGRADECIMIENTOS / DESPEDIDAS (regex)
# ============================================

SALUDOS_REGEX = re.compile(
    r"\b(hola|buenos dias|buen día|buenas tardes|buenas noches|que tal|qué tal)\b",
    re.IGNORECASE,
)
AGRADECIMIENTOS_REGEX = re.compile(
    r"\b(gracias|muchas gracias|te lo agradezco|agradecido|se agradece)\b",
    re.IGNORECASE,
)
DESPEDIDAS_REGEX = re.compile(
    r"\b(adios|adiós|hasta luego|nos vemos|chao|chau|hasta pronto)\b",
    re.IGNORECASE,
)


def handle_greetings(message: str) -> Optional[str]:
    """
    Responde a saludos, agradecimientos o despedidas usando expresiones regulares.
    Devuelve un string si detecta algo, o None si no aplica.
    """
    msg = message.lower()

    if re.search(SALUDOS_REGEX, msg):
        return (
            "¡Hola! Soy el asistente virtual de CIMA Market. "
            "¿En qué puedo ayudarte hoy?"
        )
    if re.search(AGRADECIMIENTOS_REGEX, msg):
        return "¡Con gusto! Si necesitas algo más sobre inventario o productos, aquí estoy."
    if re.search(DESPEDIDAS_REGEX, msg):
        return "¡Hasta luego! Que tengas un buen día."

    return None

# ============================================
# 4. RAG PARA FAQS
# ============================================

FAQ_TEXTS = [
    (
        "¿Cuál es el horario de atención?",
        "Nuestro horario de atención es de lunes a domingo, de 7h00 a 21h00."
    ),
    (
        "¿Dónde está ubicado el minimercado?",
        "Estamos ubicados en el barrio central, cerca del parque principal."
    ),
    (
        "¿Ofrecen servicio a domicilio?",
        "Sí, ofrecemos servicio a domicilio dentro del barrio con un costo adicional pequeño."
    ),
    (
        "¿Qué tipos de productos venden?",
        "Vendemos abarrotes, bebidas, productos de limpieza y algunos productos de cuidado personal."
    ),
    (
        "¿Puedo pagar con tarjeta?",
        "Sí, aceptamos pago en efectivo y tarjeta de débito/crédito."
    ),
]

faq_docs = [
    Document(
        page_content=f"Pregunta: {q}\nRespuesta: {a}",
        metadata={"tipo": "faq"}
    )
    for q, a in FAQ_TEXTS
]

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
faq_vectorstore = FAISS.from_documents(faq_docs, embeddings)
faq_retriever = faq_vectorstore.as_retriever(search_kwargs={"k": 3})

llm_rag = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)

faq_prompt = PromptTemplate(
    template=(
        "Eres el asistente de un minimercado. Usa solo la información del contexto "
        "para responder la pregunta del cliente.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta del cliente: {question}\n\n"
        "Respuesta en español, breve y clara:"
    ),
    input_variables=["context", "question"],
)


def answer_faq(question: str) -> str:
    """Responde FAQs usando RAG."""
    docs = faq_retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in docs)
    prompt_text = faq_prompt.format(context=context_text, question=question)
    res = llm_rag.invoke(prompt_text)
    return res.content.strip()

# ============================================
# 5. FUNCIONES DE NEGOCIO (stock, CSV)
# ============================================

def predict_stock_one(product_code: str, date: str) -> Dict[str, Any]:
    params = {
        "fecha": date,
        "lead_time_days": 5,
        "service_z": 1.28,
    }
    url = f"{BACKEND_URL}/recomendacion/{product_code}"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def predict_stock_all(date: str) -> List[Dict[str, Any]]:
    params = {
        "fecha": date,
        "lead_time_days": 5,
        "service_z": 1.28,
    }
    url = f"{BACKEND_URL}/recomendaciones"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("items", [])


def add_new_products_csv(csv_path: str) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/upload-csv/"
    with open(csv_path, "rb") as f:
        files = {"file": (os.path.basename(csv_path), f, "text/csv")}
        resp = requests.post(url, files=files)
    resp.raise_for_status()
    return resp.json()

# ============================================================
# 5.1 NUEVA FUNCIÓN 1: INVENTARIO POR CATEGORÍAS
# ============================================================

def infer_category(item: Dict[str, Any]) -> str:
    """
    Intenta inferir la categoría del producto a partir del product_id o el nombre.
    Ajusta las reglas según tus códigos reales (GRA, LAC, BEB, LIM, etc.).
    """
    pid = str(item.get("product_id", "")).upper()
    name = str(item.get("product_name", "")).lower()

    if pid.startswith("GRA") or "arroz" in name:
        return "Granos y cereales"
    if pid.startswith("LAC") or "leche" in name or "lácteos" in name or "lacteos" in name:
        return "Lácteos"
    if pid.startswith("BEB") or "agua" in name or "gaseosa" in name or "jugo" in name:
        return "Bebidas"
    if pid.startswith("END") or "azúcar" in name or "azucar" in name:
        return "Endulzantes"
    if pid.startswith("LIM") or "detergente" in name or "jabón" in name or "jabon" in name:
        return "Limpieza"
    return "Otros"


def inventory_value_by_category(date: str) -> str:
    """
    Nueva función 1:
    Calcula el 'valor' del inventario por categoría para una fecha dada,
    usando la cantidad de unidades (current_stock) como medida.
    """
    items = predict_stock_all(date)
    if not items:
        return f"No se encontraron datos de inventario para la fecha {date}."

    summary: Dict[str, Dict[str, float]] = {}

    for it in items:
        cat = infer_category(it)
        cur = float(it.get("current_stock", 0.0))

        if cat not in summary:
            summary[cat] = {"total_unidades": 0.0, "num_productos": 0}

        summary[cat]["total_unidades"] += cur
        summary[cat]["num_productos"] += 1

    lines = [f"Inventario por categorías para {date} (en unidades):"]
    for cat, info in summary.items():
        lines.append(
            f"- {cat}: {info['total_unidades']:.0f} unidades en "
            f"{info['num_productos']} productos distintos."
        )

    lines.append(
        "Este resumen muestra cuántas unidades tienes por categoría. "
        "Puedes usarlo para decidir en qué tipo de productos estás más cargado."
    )
    return "\n".join(lines)

# ============================================================
# 5.2 NUEVA FUNCIÓN 2: TOP 3 PRODUCTOS MÁS VENDIDOS
# ============================================================

def top3_best_sellers(date: str) -> str:
    """
    Nueva función 2:
    Muestra el top 3 de productos con mayor demanda diaria estimada
    (predicted_demand_1day) para una fecha dada.
    """
    items = predict_stock_all(date)
    if not items:
        return f"No se encontraron datos de inventario para la fecha {date}."

    valid = [
        it for it in items
        if it.get("predicted_demand_1day", None) is not None
    ]
    if not valid:
        return f"No hay información de demanda estimada para la fecha {date}."

    top = sorted(
        valid,
        key=lambda it: it.get("predicted_demand_1day", 0.0),
        reverse=True
    )[:3]

    lines = [f"Top 3 productos con mayor demanda estimada para {date}:"]
    for i, it in enumerate(top, start=1):
        name = it.get("product_name", it.get("product_id"))
        pid = it.get("product_id")
        dem = it.get("predicted_demand_1day", 0.0)
        cur = it.get("current_stock", 0.0)
        lines.append(
            f"{i}. {name} ({pid}): demanda diaria estimada {dem:.2f} unidades, "
            f"stock actual {cur:.0f}."
        )

    lines.append(
        "Estos productos son los que más se venden según la predicción del modelo, "
        "por lo que conviene darles prioridad en los pedidos y en la exhibición."
    )
    return "\n".join(lines)

# ============================================
# 6. FUNCTION MATCHER
# ============================================

def function_matcher(user_query: str) -> str:
    """
    Devuelve el nombre lógico de la función que se debería usar.
    Posibles retornos:
      - 'SALUDO'
      - 'FAQ'
      - 'PREDICCION_UN_PRODUCTO'
      - 'PREDICCION_TODOS'
      - 'AGREGAR_REGISTROS'
      - 'INVENTARIO'
      - 'MAX_VENDIDOS'
      - 'DESCONOCIDO'
    """
    text = user_query.lower()

    # saludos / despedidas
    if handle_greetings(text) is not None:
        return "SALUDO"

    # primero: todos los productos (para que no se confunda con uno solo)
    if "stock de todos" in text or "todos los productos" in text:
        return "PREDICCION_TODOS"

    # FAQs típicas
    if any(pal in text for pal in ["horario", "dirección", "direccion", "ubicado", "domicilio", "tarjeta"]):
        return "FAQ"

    # stock de un producto
    if "stock" in text and ("producto" in text or "código" in text or "codigo" in text):
        return "PREDICCION_UN_PRODUCTO"

    # agregar / registrar productos
    if any(p in text for p in ["agregar productos", "nuevos registros", "subir csv"]):
        return "AGREGAR_REGISTROS"

    # NUEVA FUNCIÓN 1: inventario valorizado por categorías
    if any(p in text for p in ["inventario valorizado", "valor del inventario", "por categorías", "por categorias"]):
        return "INVENTARIO"

    # NUEVA FUNCIÓN 2: top 3 más vendidos
    if any(p in text for p in ["top 3", "top3", "más vendidos", "mas vendidos", "productos más vendidos"]):
        return "MAX_VENDIDOS"

    return "DESCONOCIDO"

# ============================================
# 7. ORQUESTADOR PRINCIPAL
# ============================================

def chatbot_answer(user_query: str) -> str:
    """
    Orquesta toda la lógica:
    1. Si la pregunta contiene palabras típicas de FAQ (horario, dirección, etc.),
       responde usando RAG (answer_faq), aunque tenga un saludo.
    2. Si el mensaje es un saludo/agradecimiento/despedida CORTO, responde smalltalk.
    3. Si no, usa function_matcher para decidir la función de negocio o una respuesta genérica.
    """
    lower = user_query.lower()

    # 1) Si parece FAQ, vamos directo al RAG
    faq_keywords = [
        "horario", "hora de atención", "hora de atencion",
        "dirección", "direccion",
        "ubicado", "ubicación", "ubicacion",
        "domicilio", "servicio a domicilio",
        "tarjeta", "pago", "pagar con"
    ]
    if any(k in lower for k in faq_keywords):
        return answer_faq(user_query)

    # 2) Smalltalk SOLO si es un mensaje corto (tipo "hola", "buenas", etc.)
    greeting_resp = handle_greetings(user_query)
    if greeting_resp is not None and len(lower.split()) <= 4:
        return greeting_resp

    # 3) Decidir función
    fn_type = function_matcher(user_query)

    if fn_type == "SALUDO":
        # Por si llega aquí, devolvemos smalltalk básico
        sm = handle_greetings(user_query)
        return sm or "¡Hola! Soy el asistente de CIMA Market."

    if fn_type == "FAQ":
        return answer_faq(user_query)

    # ---------- PREDICCIÓN UN PRODUCTO ----------
    if fn_type == "PREDICCION_UN_PRODUCTO":
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_prediction_params",
                    "description": (
                        "Extrae el código de producto y la fecha de la pregunta "
                        "del usuario para hacer una predicción de stock."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_code": {
                                "type": "string",
                                "description": "Código del producto, por ejemplo ACE001",
                            },
                            "date": {
                                "type": "string",
                                "description": "Fecha en formato YYYY-MM-DD",
                            },
                        },
                        "required": ["product_code", "date"],
                    },
                },
            }
        ]

        first = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente que ayuda a extraer parámetros para "
                        "predecir el stock de un producto en un minimercado."
                    ),
                },
                {"role": "user", "content": user_query},
            ],
            tools=tools,
            tool_choice="auto",
        )

        tool_calls = first.choices[0].message.tool_calls
        if not tool_calls:
            return "No pude identificar el código de producto o la fecha. Por favor, indícalos claramente."

        args = json.loads(tool_calls[0].function.arguments)

        product_code = args.get("product_code")
        date = args.get("date")

        if not product_code or not date:
            return "Me faltan datos (código de producto o fecha) para hacer la predicción."

        try:
            resp = predict_stock_one(product_code, date)
            rec = resp["recommended_order"]
            stock = resp["current_stock"]
            target = resp["target_stock"]
            name = resp.get("product_name", product_code)
            return (
                f"Para el producto {name} ({product_code}) en la fecha {date}, "
                f"se recomienda pedir {rec} unidades. "
                f"El stock actual es {stock} y el objetivo es {target}."
            )
        except Exception as e:
            return f"Ocurrió un error al consultar el modelo de stock: {e}"

    # ---------- PREDICCIÓN TODOS LOS PRODUCTOS ----------
    if fn_type == "PREDICCION_TODOS":
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_date_param",
                    "description": "Extrae la fecha en formato YYYY-MM-DD de la pregunta.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Fecha en formato YYYY-MM-DD",
                            }
                        },
                        "required": ["date"],
                    },
                },
            }
        ]

        first = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extrae la fecha de la pregunta del usuario.",
                },
                {"role": "user", "content": user_query},
            ],
            tools=tools,
            tool_choice="auto",
        )
        tool_calls = first.choices[0].message.tool_calls
        if not tool_calls:
            return "No pude identificar la fecha para la predicción global."

        args = json.loads(tool_calls[0].function.arguments)
        date = args.get("date")
        if not date:
            return "No pude extraer una fecha válida."

        try:
            items = predict_stock_all(date)
            if not items:
                return f"No se encontraron recomendaciones para la fecha {date}."

            top = sorted(items, key=lambda r: r["recommended_order"], reverse=True)[:5]
            lines = [f"Resumen de productos con recomendación de compra para {date}:"]
            for it in top:
                lines.append(
                    f"- {it['product_name']} ({it['product_id']}): "
                    f"pedir {it['recommended_order']} unidades "
                    f"(stock actual {it['current_stock']}, objetivo {it['target_stock']})."
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Ocurrió un error al consultar el modelo de stock global: {e}"

    # ---------- AGREGAR REGISTROS ----------
    if fn_type == "AGREGAR_REGISTROS":
        return (
            "Para agregar nuevos registros de productos, sube un archivo CSV al sistema. "
            "Si estás en el cuaderno de Jupyter, puedes usar la función "
            "`add_new_products_csv('ruta/al/archivo.csv')` para enviarlo a la API."
        )

    # ---------- INVENTARIO POR CATEGORÍAS ----------
    if fn_type == "INVENTARIO":
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_date_param",
                    "description": "Extrae la fecha en formato YYYY-MM-DD de la pregunta.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Fecha en formato YYYY-MM-DD",
                            }
                        },
                        "required": ["date"],
                    },
                },
            }
        ]

        first = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extrae la fecha de la pregunta del usuario.",
                },
                {"role": "user", "content": user_query},
            ],
            tools=tools,
            tool_choice="auto",
        )
        tool_calls = first.choices[0].message.tool_calls
        if not tool_calls:
            return "No pude identificar la fecha para calcular el inventario por categorías."

        args = json.loads(tool_calls[0].function.arguments)
        date = args.get("date")
        if not date:
            return "No pude extraer una fecha válida."

        try:
            return inventory_value_by_category(date)
        except Exception as e:
            return f"Ocurrió un error al calcular el inventario por categorías: {e}"

    # ---------- TOP 3 MÁS VENDIDOS ----------
    if fn_type == "MAX_VENDIDOS":
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_date_param",
                    "description": "Extrae la fecha en formato YYYY-MM-DD de la pregunta.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Fecha en formato YYYY-MM-DD",
                            }
                        },
                        "required": ["date"],
                    },
                },
            }
        ]

        first = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extrae la fecha de la pregunta del usuario.",
                },
                {"role": "user", "content": user_query},
            ],
            tools=tools,
            tool_choice="auto",
        )
        tool_calls = first.choices[0].message.tool_calls
        if not tool_calls:
            return "No pude identificar la fecha para el top 3 de productos más vendidos."

        args = json.loads(tool_calls[0].function.arguments)
        date = args.get("date")
        if not date:
            return "No pude extraer una fecha válida."

        try:
            return top3_best_sellers(date)
        except Exception as e:
            return f"Ocurrió un error al generar el top 3 de productos más vendidos: {e}"

    # ---------- RESPUESTA GENÉRICA ----------
    generic = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres el asistente virtual de un minimercado llamado CIMA Market. "
                    "Responde en español de forma amable y breve."
                ),
            },
            {"role": "user", "content": user_query},
        ],
    )
    return generic.choices[0].message.content.strip()
