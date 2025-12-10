# ============================================
# 1. IMPORTS Y CONFIGURACIÓN
# ============================================
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

import os
import re
import json
from typing import List, Dict, Any, Optional

import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

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
    r"\b(hola|buenos dias|buen día|buen dia|buenas tardes|buenas noches|que tal|qué tal)\b",
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
    """Responde FAQs usando RAG y deja logs de qué FAQ se usó."""
    docs = faq_retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in docs)

    logging.info("[RAG][FAQ] Pregunta usuario: %s", question)

    for idx, d in enumerate(docs, start=1):
        first_line = d.page_content.split("\n")[0]
        if first_line.lower().startswith("pregunta:"):
            faq_q = first_line.split(":", 1)[1].strip()
        else:
            faq_q = first_line.strip()

        logging.info("[RAG][FAQ] Top %d -> %s", idx, faq_q)

    if docs:
        first_line = docs[0].page_content.split("\n")[0]
        if first_line.lower().startswith("pregunta:"):
            faq_match = first_line.split(":", 1)[1].strip()
        else:
            faq_match = first_line.strip()
        logging.info("[RAG][FAQ_MATCH] FAQ seleccionada (Top 1): %s", faq_match)

    prompt_text = faq_prompt.format(context=context_text, question=question)
    res = llm_rag.invoke(prompt_text)
    return res.content.strip()


# ============================================
# 5. FUNCIONES DE NEGOCIO (stock, CSV)
# ============================================

def predict_stock_one(product_code: str, date: str) -> Dict[str, Any]:
    """Llama al backend para la recomendación de UN producto."""
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
    """Llama al backend para la recomendación de TODOS los productos."""
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
# 5.1 INVENTARIO POR CATEGORÍAS
# ============================================================

def infer_category(item: Dict[str, Any]) -> str:
    """Inferir categoría aproximada a partir de ID o nombre."""
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
    Calcula 'inventario por categorías' para una fecha,
    usando current_stock como medida.
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
# 5.2 TOP 3 PRODUCTOS MÁS VENDIDOS
# ============================================================

def top3_best_sellers(date: str) -> str:
    """
    Top 3 de productos con mayor demanda diaria estimada
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
    STRICT MODE: detecta funciones en este orden de prioridad:
    1) Saludos
    2) Predicción TODOS los productos
    3) Predicción de UN producto
    4) Top 3 más vendidos
    5) Inventario por categorías
    6) Agregar registros (CSV)
    7) FAQs
    8) Desconocido
    """
    text = user_query.lower()

    # 1) SALUDOS / AGRADECIMIENTOS / DESPEDIDAS
    if handle_greetings(text) is not None:
        fn = "SALUDO"
        logging.info("[FUNCTION_MATCHER] Query='%s' => %s", user_query, fn)
        return fn

    # 2) PREDICCIÓN DE STOCK DE TODOS LOS PRODUCTOS
    if "stock de todos" in text or "todos los productos" in text:
        fn = "PREDICCION_TODOS"
        logging.info("[FUNCTION_MATCHER] Query='%s' => %s", user_query, fn)
        return fn

    # 3) PREDICCIÓN DE STOCK DE UN PRODUCTO
    if "stock" in text and ("producto" in text or "código" in text or "codigo" in text):
        fn = "PREDICCION_UN_PRODUCTO"
        logging.info("[FUNCTION_MATCHER] Query='%s' => %s", user_query, fn)
        return fn

    # 4) TOP 3 PRODUCTOS MÁS VENDIDOS
    if any(p in text for p in [
        "top 3", "top3",
        "tres más vendidos", "tres mas vendidos",
        "productos más vendidos", "productos mas vendidos"
    ]):
        fn = "MAX_VENDIDOS"
        logging.info("[FUNCTION_MATCHER] Query='%s' => %s", user_query, fn)
        return fn

    # 5) INVENTARIO VALORIZADO / POR CATEGORÍAS
    if any(p in text for p in [
        "inventario valorizado",
        "valor del inventario",
        "inventario por categorías",
        "inventario por categorias"
    ]):
        fn = "INVENTARIO"
        logging.info("[FUNCTION_MATCHER] Query='%s' => %s", user_query, fn)
        return fn

    # 6) AGREGAR REGISTROS DESDE CSV
    if any(p in text for p in [
        "agregar productos",
        "nuevos registros",
        "subir csv",
        "cargar csv"
    ]):
        fn = "AGREGAR_REGISTROS"
        logging.info("[FUNCTION_MATCHER] Query='%s' => %s", user_query, fn)
        return fn

    # 7) FAQ / RAG
    faq_keywords = [
        "horario", "hora de atención", "hora de atencion",
        "dirección", "direccion",
        "ubicado", "ubicación", "ubicacion",
        "domicilio", "servicio a domicilio",
        "tarjeta", "pago", "pagar",
        "productos", "producto", "ofrecen", "venden"
    ]
    if any(k in text for k in faq_keywords):
        fn = "FAQ"
        logging.info("[FUNCTION_MATCHER] Query='%s' => %s", user_query, fn)
        return fn

    # 8) DESCONOCIDO
    fn = "DESCONOCIDO"
    logging.info("[FUNCTION_MATCHER] Query='%s' => %s", user_query, fn)
    return fn

# ============================================
# 7. ORQUESTADOR PRINCIPAL
# ============================================

def _extract_date_from_text(text: str) -> Optional[str]:
    """Busca fecha tipo YYYY-MM-DD en el texto."""
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    return m.group(1) if m else None


def _extract_product_code(text: str) -> Optional[str]:
    """
    Busca algo tipo 'producto ACE001' o 'código ACE001'.
    Ajustado a tu forma de preguntar en la rúbrica.
    """
    m = re.search(
        r"(?:producto|c[oó]digo)\s+([A-Za-z0-9_-]+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    return None


def chatbot_answer(user_query: str) -> str:
    """
    Orquesta toda la lógica apoyándose SIEMPRE en function_matcher
    para decidir qué hacer.
    """
    fn_type = function_matcher(user_query)

    # ---------- SALUDOS / DESPEDIDAS ----------
    if fn_type == "SALUDO":
        sm = handle_greetings(user_query)
        return sm or "¡Hola! Soy el asistente de CIMA Market."

    # ---------- FAQ CON RAG ----------
    if fn_type == "FAQ":
        return answer_faq(user_query)

    # ---------- PREDICCIÓN UN PRODUCTO ----------
    if fn_type == "PREDICCION_UN_PRODUCTO":
        date = _extract_date_from_text(user_query)
        product_code = _extract_product_code(user_query)

        if not date or not product_code:
            return (
                "Para ayudarte con la predicción necesito que indiques claramente "
                "la fecha (YYYY-MM-DD) y el código del producto. Ejemplo: "
                "«Quiero saber el stock del producto ACE001 para el 2025-03-14»."
            )

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
        date = _extract_date_from_text(user_query)
        if not date:
            return (
                "Necesito que especifiques la fecha en formato YYYY-MM-DD. "
                "Ejemplo: «Muéstrame el stock de todos los productos para el 2025-03-14»."
            )

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
            "Para agregar nuevos registros de productos, sube un archivo CSV en la sección "
            "«Cargar nuevos registros (CSV)» de la aplicación. También puedes adjuntar el CSV "
            "en este chat y escribir algo como «Actualiza el modelo con los nuevos registros» "
            "para que el sistema lo cargue y reentrene el modelo."
        )

    # ---------- INVENTARIO POR CATEGORÍAS ----------
    if fn_type == "INVENTARIO":
        date = _extract_date_from_text(user_query)
        if not date:
            return (
                "Necesito la fecha en formato YYYY-MM-DD para calcular el inventario por categorías. "
                "Ejemplo: «Muéstrame el inventario valorizado por categorías para el 2025-03-14»."
            )
        try:
            return inventory_value_by_category(date)
        except Exception as e:
            return f"Ocurrió un error al calcular el inventario por categorías: {e}"

    # ---------- TOP 3 MÁS VENDIDOS ----------
    if fn_type == "MAX_VENDIDOS":
        date = _extract_date_from_text(user_query)
        if not date:
            return (
                "Necesito la fecha en formato YYYY-MM-DD para el top 3 de productos más vendidos. "
                "Ejemplo: «Dame el top 3 de productos más vendidos para el 2025-03-14»."
            )

        try:
            return top3_best_sellers(date)
        except Exception as e:
            return f"Ocurrió un error al generar el top 3 de productos más vendidos: {e}"

    # ---------- RESPUESTA GENÉRICA ----------
    logging.info("[CHATBOT] Query desconocida, usando respuesta genérica fija")
    return (
        "Soy el asistente virtual de CIMA Market. "
        "Por ahora solo puedo ayudarte con:\n"
        "- Horario y ubicación del minimercado\n"
        "- Servicio a domicilio y formas de pago\n"
        "- Consultas de stock y predicciones de inventario\n"
        "- Resúmenes de inventario por categorías y top de más vendidos\n\n"
        "Por favor reformula tu pregunta usando alguno de estos temas."
    )
