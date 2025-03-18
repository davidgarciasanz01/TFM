# Lista de preguntas y respuestas con palabras clave para la parte de obras y recorridos
questions = [
    "¿Cuáles son las principales obras que se pueden ver en el Museo Arqueológico Nacional?",
    "¿Qué recorridos temáticos tiene el museo?",
    "¿Qué es la Dama de Elche y por qué es tan importante?",
    "¿En qué sala se exhibe el Tesoro de Guarrazar?",
    "¿El museo tiene alguna exposición sobre la antigua Roma?",
    "¿Cuáles son las obras más representativas de Grecia?",
    "¿El museo ofrece visitas guiadas centradas en una época específica?",
    "¿Qué salas están dedicadas a la cultura egipcia?",
    "¿El museo tiene una sección dedicada a la Edad Media?",
    "¿Cómo puedo participar en el recorrido Arqueología de la Muerte?"
]

# Palabras clave específicas para cada pregunta sobre obras y recorridos
keywords_for_questions = {
    "¿Cuáles son las principales obras que se pueden ver en el Museo Arqueológico Nacional?": ["principales", "obras", "Museo Arqueológico Nacional", "colección", "Dama de Elche", "Toros de Costitx"],
    "¿Qué recorridos temáticos tiene el museo?": ["recorridos", "temáticos", "museo", "historia", "cultura", "épocas"],
    "¿Qué es la Dama de Elche y por qué es tan importante?": ["Dama de Elche", "escultura", "importante", "museo", "íbera"],
    "¿En qué sala se exhibe el Tesoro de Guarrazar?": ["Tesoro de Guarrazar", "sala", "exhibición", "museo", "visigodo"],
    "¿El museo tiene alguna exposición sobre la antigua Roma?": ["exposición", "antigua Roma", "museo", "Roma", "colección"],
    "¿Cuáles son las obras más representativas de Grecia?": ["obras", "representativas", "Grecia", "museo", "colección"],
    "¿El museo ofrece visitas guiadas centradas en una época específica?": ["visitas guiadas", "época", "específica", "museo", "historia"],
    "¿Qué salas están dedicadas a la cultura egipcia?": ["salas", "cultura egipcia", "Egipto", "museo", "colección"],
    "¿El museo tiene una sección dedicada a la Edad Media?": ["sección", "Edad Media", "museo", "historia", "medieval"],
    "¿Cómo puedo participar en el recorrido Arqueología de la Muerte?": ["recorrido", "Arqueología de la Muerte", "museo", "temático", "actividades"]
}

from LLM import info_run_guide_llm  # Importa las funciones del LLM general
from langchain_community.chat_message_histories import ChatMessageHistory
from LLM_Guia import run_guide_llm

# Función para evaluar la respuesta generada por las palabras clave
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

# Inicializar historial vacío
chat_history = []

# Evaluar cada respuesta generada
for question in questions:
    # Obtener las palabras clave específicas para cada pregunta
    expected_keywords = keywords_for_questions[question]
    
    # Generar la respuesta usando el LLM
    generated_response = run_guide_llm(question, chat_history)
    
    # Evaluar la respuesta
    evaluation = evaluate_response(generated_response, expected_keywords)
    
    # Imprimir la evaluación
    print(f"Evaluación para la pregunta '{question}':")
    print(f"Palabras clave: {', '.join(expected_keywords)}")

    print(f"Respuesta Generada: {generated_response}")

    print(f"Evaluación: {evaluation}")
    print("-" * 50)
