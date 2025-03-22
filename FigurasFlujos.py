import graphviz

# Crear el diagrama para la inserción de embeddings en Pinecone
dot1 = graphviz.Digraph('Inserción_Pinecone', format='png')

dot1.node('A', 'Obtención de Datos', shape='box', style='filled', fillcolor='lightblue')
dot1.node('B', 'Preprocesamiento de Datos', shape='box', style='filled', fillcolor='lightblue')
dot1.node('C', 'Generación de Embeddings', shape='box', style='filled', fillcolor='lightblue')
dot1.node('D', 'Almacenamiento en Pinecone', shape='box', style='filled', fillcolor='lightblue')

dot1.edge('A', 'B', label='Limpieza y normalización')
dot1.edge('B', 'C', label='Generación con OllamaEmbeddings')
dot1.edge('C', 'D', label='Inserción en Pinecone con metadatos')

# Crear el diagrama para el flujo completo del sistema
dot2 = graphviz.Digraph('Flujo_Sistema', format='png')

dot2.node('1', 'Usuario introduce consulta en Streamlit', shape='box', style='filled', fillcolor='lightgreen')
dot2.node('2', 'Retriever accede a Pinecone', shape='box', style='filled', fillcolor='lightblue')
dot2.node('3', 'Recuperación de documentos más cercanos', shape='box', style='filled', fillcolor='lightblue')
dot2.node('4', 'LLM (LLaMa 3.3) genera respuesta', shape='box', style='filled', fillcolor='lightblue')
dot2.node('5', 'Respuesta mostrada en Streamlit', shape='box', style='filled', fillcolor='lightgreen')

dot2.edge('1', '2', label='Consulta enviada')
dot2.edge('2', '3', label='Búsqueda en Pinecone')
dot2.edge('3', '4', label='Envío a LLaMa 3.3')
dot2.edge('4', '5', label='Presentación de respuesta')

# Renderizar y guardar los diagramas
dot1_path = "c:/Users/David/Desktop/MSC DATA SCIENCE/2 CUATRIMESTRE/TFM/MAN/insercion_pinecone"
dot2_path = "c:/Users/David/Desktop/MSC DATA SCIENCE/2 CUATRIMESTRE/TFM/MAN/flujo_sistema"
print(dot1_path)
print(dot1_path)
dot1.render(dot1_path)
dot2.render(dot2_path)

dot1_path + ".png", dot2_path + ".png"
