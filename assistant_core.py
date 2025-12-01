# assistant_core.py
import re
from dataclasses import dataclass

# ============================================================
# 1) Datos de la empresa ficticia (nicho)
# ============================================================

@dataclass
class CompanyInfo:
    name: str
    niche: str
    founded_year: int
    description: str
    milestones: list[str]
    products: list[str]
    services: list[str]
    opening_hours: str
    contact: str

COMPANY = CompanyInfo(
    name="CIMA Market",
    niche="Minimercado de barrio enfocado en productos de consumo diario",
    founded_year=2015,
    description=(
        "CIMA Market es un minimercado de barrio que ofrece productos de "
        "consumo diario como abarrotes, lácteos, bebidas y artículos de limpieza. "
        "Está orientado a familias del sector y pequeños negocios de la zona."
    ),
    milestones=[
        "2015: Apertura del primer local en el barrio La Floresta.",
        "2018: Implementación de sistema de inventario digital.",
        "2022: Integración de un modelo de predicción de demanda para optimizar pedidos.",
    ],
    products=[
        "Abarrotes (arroz, azúcar, fideos, aceite vegetal, etc.)",
        "Lácteos (leche, yogurt, queso fresco)",
        "Bebidas (jugos, gaseosas, agua embotellada)",
        "Snacks (galletas, chocolates, papas fritas)",
        "Artículos de limpieza (detergente, limpiavidrios, lavavajillas)",
    ],
    services=[
        "Venta al detalle en tienda física",
        "Preparación de pedidos para recoger en tienda",
        "Entrega a domicilio en el barrio (según disponibilidad)",
    ],
    opening_hours="Lunes a domingo, de 7h00 a 21h00.",
    contact="Teléfono: 099-999-9999 | Correo: contacto@cimamarket.com",
)


def get_company_overview() -> str:
    """
    Devuelve una descripción general de la empresa para usar en el LLM
    o como respuesta directa en el chatbot.
    """
    text = (
        f"{COMPANY.name} es un {COMPANY.niche}, fundado en {COMPANY.founded_year}. "
        f"{COMPANY.description}\n\n"
        f"Algunos hitos importantes:\n"
    )
    for m in COMPANY.milestones:
        text += f"- {m}\n"

    text += "\nProductos principales:\n"
    for p in COMPANY.products:
        text += f"- {p}\n"

    text += "\nServicios:\n"
    for s in COMPANY.services:
        text += f"- {s}\n"

    text += f"\nHorario de atención: {COMPANY.opening_hours}\n"
    text += f"Contacto: {COMPANY.contact}"
    return text


# ============================================================
# 2) Saludos, agradecimientos y despedidas (regex)
# ============================================================

SALUDOS_REGEX = re.compile(
    r"\b(hola|buenas|buenos dias|buen día|buenas tardes|buenas noches|qué tal|que tal)\b",
    re.IGNORECASE
)

AGRADECIMIENTOS_REGEX = re.compile(
    r"\b(gracias|muchas gracias|te agradezco|se agradece|muy amable)\b",
    re.IGNORECASE
)

DESPEDIDAS_REGEX = re.compile(
    r"\b(adios|adiós|hasta luego|nos vemos|chao|chau|me despido)\b",
    re.IGNORECASE
)


def handle_smalltalk(user_text: str) -> str | None:
    """
    Detecta si el texto del usuario es un saludo, agradecimiento o despedida
    y devuelve una respuesta adecuada. Si no detecta nada, devuelve None.
    """
    text = user_text.strip().lower()

    if SALUDOS_REGEX.search(text):
        return (
            f"¡Hola! Soy el asistente virtual de {COMPANY.name}. "
            f"¿En qué puedo ayudarte hoy?\n"
            f"Nuestro horario de atención es: {COMPANY.opening_hours}"
        )

    if AGRADECIMIENTOS_REGEX.search(text):
        return "¡Con gusto! Si necesitas algo más sobre productos o inventario, estoy aquí para ayudarte."

    if DESPEDIDAS_REGEX.search(text):
        return "¡Hasta luego! Gracias por contactar con CIMA Market."

    return None


# ============================================================
# 3) Router básico para probar (sin RAG ni FunctionMatcher todavía)
# ============================================================

def process_query_basic(user_text: str) -> str:
    """
    Router básico:
    1) Intenta responder saludos/agradecimientos/despedidas.
    2) Si el usuario pide información de la empresa, devuelve overview.
    3) Si no, devuelve un mensaje genérico (por ahora).
    """
    # 1) Smalltalk
    smalltalk_resp = handle_smalltalk(user_text)
    if smalltalk_resp is not None:
        return smalltalk_resp

    # 2) Preguntas sobre la empresa (muy simple por ahora)
    if "empresa" in user_text.lower() or "cima market" in user_text.lower():
        return get_company_overview()

    # 3) Fallback genérico (luego aquí enchufamos RAG / FunctionMatcher)
    return (
        "Por ahora solo puedo responder saludos, agradecimientos, despedidas "
        "y algunas preguntas básicas sobre CIMA Market. "
        "Pronto integraré funciones avanzadas de inventario."
    )


if __name__ == "__main__":
    while True:
        q = input("Tú: ")
        if q.lower().strip() in ("salir", "exit", "quit"):
            break
        print("Bot:", process_query_basic(q))
