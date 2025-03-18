from LLM_Guia import run_guide_llm
from langchain_community.chat_message_histories import ChatMessageHistory
from LLM import info_run_guide_llm  # Importa las funciones del LLM general

# Función para evaluar la respuesta generada
def evaluate_response(generated_response, expected_keywords):
    # Convertir la respuesta generada a minúsculas para hacer la comparación insensible a mayúsculas/minúsculas
    generated_response_lower = generated_response.lower()
    
    # Verificar si las palabras clave de la respuesta esperada están presentes en la respuesta generada
    missing_keywords = []
    for keyword in expected_keywords:
        if keyword.lower() not in generated_response_lower:
            missing_keywords.append(keyword)
    
    # Si faltan palabras clave, la respuesta es incorrecta
    if missing_keywords:
        return f"Incorrecta. Faltan las palabras clave: {', '.join(missing_keywords)}."
    else:
        return "Correcta. Todas las palabras clave están presentes."

# Lista de preguntas y palabras clave específicas para cada pregunta
questions = [
    "¿Cuáles son los horarios del museo?",
    "¿El museo tiene acceso para personas con movilidad reducida?",
    "¿Cuánto cuesta la entrada general?",
    "¿El museo abre los días festivos?",
    "¿A qué hora cierra el museo hoy?",
    "¿Puedo obtener entradas en línea?",
    "¿Cuál es el precio de la entrada para un grupo de 10 personas?",
    "¿El museo tiene estacionamiento disponible?",
    "¿Ofrecen audioguías en otros idiomas?",
    "¿Dónde está ubicado el Museo Arqueológico Nacional?",
    "¿Cómo llego al museo desde la estación de tren más cercana?",
    "¿Puedo organizar una visita privada al museo?",
    "¿Se pueden hacer fotos dentro del museo?",
    "¿Hay actividades para niños?"
]

# Palabras clave específicas para cada pregunta
keywords_for_questions = {
    "¿Cuáles son los horarios del museo?": ["horarios", "museo", "hora"],
    "¿El museo tiene acceso para personas con movilidad reducida?": ["acceso", "movilidad reducida", "accesibilidad"],
    "¿Cuánto cuesta la entrada general?": ["precio", "entrada", "general", "costo"],
    "¿El museo abre los días festivos?": ["festivos", "abierto", "días festivos"],
    "¿A qué hora cierra el museo hoy?": ["hora", "cierra", "hoy"],
    "¿Puedo obtener entradas en línea?": ["entradas", "en línea", "compra"],
    "¿Cuál es el precio de la entrada para un grupo de 10 personas?": ["precio", "entrada", "grupo", "10 personas"],
    "¿El museo tiene estacionamiento disponible?": ["estacionamiento", "disponible", "museo"],
    "¿Ofrecen audioguías en otros idiomas?": ["audioguías", "otros idiomas", "ofrecen"],
    "¿Dónde está ubicado el Museo Arqueológico Nacional?": ["ubicación", "museo", "dirección"],
    "¿Cómo llego al museo desde la estación de tren más cercana?": ["llegar", "estación de tren", "cerca", "museo"],
    "¿Puedo organizar una visita privada al museo?": ["visita privada", "organizar", "museo"],
    "¿Se pueden hacer fotos dentro del museo?": ["fotos", "permitido", "museo"],
    "¿Hay actividades para niños?": ["actividades", "niños", "museo"]
}

chat_history = ChatMessageHistory()  # Inicializar historial vacío

# Evaluar cada respuesta generada
for question in questions:
    # Obtén las palabras clave específicas para cada pregunta
    expected_keywords = keywords_for_questions[question]
    
    # Generar la respuesta usando el LLM
    generated_response = info_run_guide_llm(question, chat_history)
    
    # Evaluar la respuesta
    evaluation = evaluate_response(generated_response, expected_keywords)
    
    # Imprimir la evaluación
    print(f"Evaluación para la pregunta '{question}':")
    print(f"Palabras clave: {', '.join(expected_keywords)}")

    print(f"Respuesta Generada: {generated_response}")

    print(f"Evaluación: {evaluation}")
    print("-" * 50)
