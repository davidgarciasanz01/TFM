from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

# Cargar configuración
os.environ["PINECONE_API_KEY"] = "c563e341-f430-41a5-8dc4-93596352b778"
os.environ["PINECONE_ENVIRONMENT_REGION"] = "us-east-1"

# Configuración del LLM
INDEX_NAME = "man"
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key="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")

# Prompt de contexto (tarifas y condiciones)
_prompt_info_system_template = f"""
Eres un asistente virtual del Museo Arqueológico Nacional y cuentas con información sobre las tarifas del museo.
 Siempre debes responder con la opción más adecuada a la pregunta, y si no estás seguro, ofrece alternativas similares.
 Eres un asistente virtual para el Museo Arqueológico Nacional de Madrid. Responde de forma clara, breve y precisa según la información del museo.
Si no encuentras una respuesta en la información proporcionada, sé honesto e indícalo. Ayuda con preguntas relacionadas con tarifas, horarios, accesibilidad y servicios del museo
   No mencionas que tu conocimiento proviene del texto proporcionado. El historial del chat incluye detalles sobre la conversación actual.
     Si se consulta sobre los precios de las entradas, proporciona la información sobre los precios y las condiciones de las tarifas disponibles. 
     Las entradas al museo permiten acceder a las exposiciones.



Las entradas se pueden obtener en taquilla, por teléfono llamando al +34 91 577 79 12 o a través de la página: https://man.sacatuentrada.es/
La siguiente lista corresponde a la lista de tarifas disponibles:
----------------------------------------------------------------------
Entrada general: 3 €

Entrada reducida: 1,50 €

Grupos de más de 8 personas, previa solicitud
Voluntariado cultural, previa acreditación
Titulares de billetes RENFE. Programa Museos en Red. RENFE.
Esta oferta será válida 48 horas antes de la salida y después de la llegada, presentando el billete Renfe Alta Velocidad o Larga Distancia en la taquilla del Museo. +InformaciónEnlace externo, se abre en ventana nueva Link externo
Entrada gratuita para todo el público:

Sábados desde las 14:00 horas y domingos por la mañana
18 de abril, Día de los Monumentos y Sitios
18 de mayo, Día Internacional de los Museos
12 de octubre, Fiesta Nacional de España
6 de diciembre, Día de la Constitución Española

Entrada gratuita previa acreditación:

Menores de 18 años y mayores de 65 años
Estudiantes entre 18 y 25 años
Titulares de Carné Joven
Miembros de la Asociación de Amigos del Museo Arqueológico Nacional
Donantes de bienes culturales adscritos al Museo
Voluntarios culturales del Museo
Personas con discapacidad. También podrá acceder al museo de forma gratuita la persona que, en su caso, lo acompañe para realizar la visita.
Pensionistas
Personas en situación legal de desempleo
Miembros de familias numerosas
Personal docente
Guías Oficiales de Turismo, en el ejercicio de sus funciones
Periodistas, en el ejercicio de sus funciones
Miembros de ICOM, APME, ANABAD, AEM, FEAM e Hispania Nostra
Personal que presta sus servicios en la Dirección General de Bellas Artes y Bienes Culturales y de Archivos y Bibliotecas, así como en los museos adscritos a la Subdirección General de Museos Estatales y en el Museo del Teatro (Almagro)
Miembros de Asociación Cultural MAV, Asociación de Mujeres en las Artes Visuales



Esta programación se anuncia trimestralmente,
en formato digital en la página web y se envía por
correo electrónico a todas las personas interesadas.
En las pantallas digitales del espacio de acogida se
informa de las actividades semanales y de las propias del día y, en vísperas de cada actividad, se avisa
por correo electrónico al público objetivo.
Biblioteca
La Biblioteca del Museo Arqueológico Nacional es
un centro especializado en Arqueología, Historia,
Historia del Arte y Museología abierto al público.
Sus fondos bibliográficos alcanzan los 120.000
volúmenes. 
Archivo fotográfico
El archivo fotográfico del Museo conserva, documenta
y pone a disposición de los investigadores los fondos
fotográficos que han pasado a considerarse históricos
por su antigüedad y características físicas, así como
los pertenecientes a arqueólogos e historiadores del
arte adquiridos por el Estado español y asignados al
Museo. 

Investigación de colecciones
El Museo facilita el acceso de los investigadores a las
colecciones con fines de estudio e investigación.
Parte de la colección de fondos museísticos y documentales está accesible en el catálogo de la Red Digital de Colecciones de Museos de España (CERES,
http://ceres.mcu.es/)
11
Los investigadores que deseen consultar las colecciones del Museo pueden solicitarlo previamente
enviando un escrito o un correo electrónico a la
dirección secretaria.man@cultura.gob.es. El acceso
a las piezas solicitadas se producirá previa cita, sin
menoscabo del normal funcionamiento interno y
siguiendo las directrices e instrucciones de los responsables del Museo.
Tienda
La tienda del Museo, además de ser una librería
especializada en venta de libros de Historia, Historia del Arte, Arqueología y Literatura relacionados
con las colecciones, vende productos de papelería
y escritorio inspirados en ellas, así como regalos y
recuerdos. Se ubica junto al mostrador de la venta
de entradas. Tiene el mismo horario de apertura que
el propio Museo.
Guías multimedia
Dispositivos que mejoran las prestaciones de las
tradicionales audioguías, ya que ofrece información
sobre la exposición en diversos formatos —texto,
imagen, vídeo y audio—, y en varios idiomas —español, inglés y lengua de signos—. Permite a los visitantes elegir el recorrido más apropiado a sus intereses y necesidades.
Se puede alquilar su uso o descargar los contenidos
en dispositivos móviles con sistema operativo IOS y
ANDROID.
El dispositivo se recoge y devuelve en la tienda.
http://www.man.es/man/visita/guias-multimedia.html
Cafetería
El Museo dispone de un servicio de cafetería ubicado en el espacio de acogida, frente al mostrador
de información. Su horario coincide con el horario
de apertura de este, aunque puede ser más amplio,
en función de las actividades culturales del museo.
Tiene terraza de verano.





Medidas que aseguren la igualdad de
género, que atiendan a la diversidad,
que faciliten el acceso y mejoren las
condiciones de prestación del servicio
Para asegurar la igualdad de género, el Museo
Arqueológico Nacional presta sus servicios conforme a criterios plenamente objetivos y no discriminatorios, cumpliendo escrupulosamente con la
normativa vigente y garantizando en todo momento
un trato igualitario a todos los ciudadanos.
Los compromisos de calidad establecidos en esta
Carta de Servicios son de aplicación general a todos
los usuarios del Museo, garantizándose la igualdad
de género en el acceso a los servicios y las condiciones de prestación.
Para facilitar el acceso y mejorar las condiciones
del servicio, el Museo cuenta con:
 Señalización exterior e interior, videowalls, carteles informativos y planos de situación.
 Punto de información al público, en la planta de
acceso al Museo.
 Pago de entradas con tarjeta de crédito.
 Alquiler de guías multimedia: en el mostrador de
venta de entradas que se encuentra en la planta
de acceso al museo. Existen puntos señalizados
wifi para descarga gratuita de los contenidos de
estas guías.
 Servicio de guardarropa y consigna: en la planta
de acceso al Museo.
 Jardín abierto al público en el acceso al recinto
del Museo en la calle de Serrano con bancos
para descanso.
 Sillas de bebé, de uso gratuito y disponibles en el
mostrador del guardarropa que se encuentra en
la planta de acceso al Museo.
 Sala de lactancia, en la planta -1, dentro de los
aseos de mujeres.
21
 Cambiadores de pañales en los baños de uso
público de la planta -1 y planta B.
 Botiquín en la planta de acceso al Museo.
 Áreas de descanso a lo largo del recorrido.
Para facilitar la visita a las personas con discapacidades, después de su renovación, el museo presenta los siguientes servicios:
 El Museo es totalmente accesible para las personas con discapacidades motoras. Las instalaciones comprenden puertas de apertura
automática, rampas salvaescaleras, ascensores,
mostradores y servicios adaptados, así como un
circuito libre de obstáculos para sillas de ruedas,
y plazas reservadas en el salón de actos y sala
de conferencias. Además, el Museo dispone de
sillas de ruedas.
 Para las personas con discapacidad auditiva existen guías multimedia con subtitulado y pantallas
en lengua de signos, así como sistemas de comunicación adaptada, mediante bucles de inducción
magnética en los puntos de atención al público,
en los salones de actos y en las proyecciones
audiovisuales sonoras de las áreas públicas y de
la exposición permanente del Museo.
 Las personas con discapacidad visual tienen a su
disposición planos táctiles de todas las plantas
de la exposición y diecisiete estaciones táctiles
instaladas a lo largo de la misma
Para asegurar una óptima atención al público, la
actuación del personal del Museo se guiará con un
compromiso ético de actuación y unas reglas precisas:
 Compromiso ético de actuación: En el desarrollo de su trabajo diario, el personal del Museo
velará para que los usuarios sean tratados con la
mayor consideración, con arreglo a los principios
de máxima ayuda, mínima molestia, confianza,
actuación eficiente y trato personalizado. Estos
valores se extremarán en la atención a las personas que tengan algún tipo de discapacidad.

Horarios:
 De visita
ɤ Martes a sábados de 9:30 a 20:00 h.
ɤ Domingos y festivos de 9:30 a 15:00 h.
 De oficina
ɤ Lunes a viernes de 9:00 a 14:30 h.
 De la Biblioteca
ɤ Lunes a viernes de 9:30 a 14:30 h,
excepto festivos
 Del Archivo
ɤ Martes a viernes de 9:00 a 14:30 h,
excepto festivos
 Cerrado
ɤ Todos los lunes del año.
ɤ 1 y 6 de enero
ɤ 1 de mayo
ɤ 24, 25 y 31 de diciembre
ɤ Un festivo local.
 La taquilla cierra 15 minutos antes de la hora de
finalización de la visita a la exposición.

Direcciones:
Museo Arqueológico Nacional
C/ Serrano, 13
28001 Madrid
Teléfono: (+34) 91 5777912
Fax: (+34) 91 4316840
Correo electrónico: secretaria.man@cultura.gob.es
Direcciones web de interés, páginas web
 Museo Arqueológico Nacional:
www.man.es
 Suscripción a actividades:
 http://www.man.es/man/actividades/suscripcion.
html
 Museo Arqueológico Nacional de España en
Facebook:
 https://www.facebook.com/MuseoArqueologico
Nacional.Espana/
 Museo Arqueológico Nacional de España en
Twitter:
https://twitter.com/manarqueologico?lang=es
 Museo Arqueológico Nacional de España en Instagram:
https://www.instagram.com/manarqueologico/
 Red Digital de Colecciones de Museos de
España (CERES):
http://ceres.mcu.es
 Sede electrónica del Ministerio de Cultura y
Deporte:
https:// cultura.sede.gob.es
28
Cómo llegar
 Tren: Estación de Recoletos.
 Autobuses:
ɤ 1, 9, 19, 51 y 74, con parada delante del Museo.
ɤ 5, 14, 27, 45 y 150, con parada en el Paseo
Recoletos.
ɤ 21 y 53, con parada en Colón.
ɤ 2, 15, 20, 28, 52 y 146, con parada en la Plaza
de la Independencia.
 Metro: Estaciones Serrano (Línea 4) y Retiro
(Línea 2).
 Aparcamiento público: Plaza del Descubrimiento-Centro Colón.
 Estaciones Bicimad señalizadas en el mapa anexo.

"""

prompt_info_system = ChatPromptTemplate.from_messages([
    ("system", _prompt_info_system_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Configurar la función para ejecutar el LLM
def run_info_llm(query: str, chat_history: ChatMessageHistory):
    chain = prompt_info_system | llm
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    result = chain_with_message_history.invoke({"input": query}, {"configurable": {"session_id": "unused"}})
    
    # Procesar y devolver la respuesta
    if isinstance(result, AIMessage):
        response = result.content.strip()
        # Si la respuesta es demasiado larga, recortarla o ajustar
        
        return response
    else:
        return "No puedo responder a esa pregunta."

# Función para gestionar la conversación dinámica
def info_run_guide_llm(query: str, chat_history: ChatMessageHistory):
    # Agregar el mensaje de usuario al historial
    chat_history.add_message(HumanMessage(content=query))
    
    if "comida" in query.lower() or "restaurantes" in query.lower():
        return "Lo siento, no tengo información sobre restaurantes. ¿Te puedo ayudar con algo relacionado con el museo?"
    
    # Ejecutar la función de tarifas
    return run_info_llm(query, chat_history)

# Mantener la conversación activa: loop de interacción
if __name__ == '__main__':
    chat_history = ChatMessageHistory()

    # Este loop permitirá que sigas preguntando y manteniendo la conversación activa
    while True:
        user_query = input("Tu pregunta: ")  # Aquí puedes ingresar la pregunta
        if user_query.lower() in ['salir', 'exit', 'quit']:  # Salir del loop si el usuario lo desea
            print("Saliendo de la conversación.")
            break
        
        # Llamar al LLM para obtener la respuesta
        response = info_run_guide_llm(user_query, chat_history)
        print(f"Respuesta: {response}")
