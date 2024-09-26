## Asistente de Salud Mental - Versión 1.0

### Descripción del Proyecto

**Asistente de Salud Mental (ASM) v1.0** es un chatbot diseñado para brindar apoyo emocional utilizando un análisis de sentimiento basado en IA para adaptar las respuestas según la entrada del usuario. El proyecto seguirá un desarrollo incremental, comenzando con funcionalidades básicas y mejorando progresivamente hasta llegar a un prototipo más estable. Cada versión será probada, y se documentarán las mejoras para futuras iteraciones.

### Funcionalidades en la Versión 1.0:
1. **Análisis de Sentimiento**: Integra un modelo de análisis de sentimientos de Hugging Face para detectar las emociones del usuario y ajustar las respuestas.
2. **Funcionalidad Básica de Chatbot**: Utiliza un conjunto de preguntas y respuestas predefinidas y un modelo LLM (Llama 2) para generar respuestas a preguntas no cubiertas por el dataset.
3. **Interfaz con Gradio**: Una interfaz web sencilla utilizando Gradio para que los usuarios interactúen con el chatbot.
4. **Autoaprendizaje**: El chatbot agrega dinámicamente nuevas preguntas y respuestas al dataset de preguntas y respuestas.

### Tecnologías Utilizadas:
- **Python**: Lenguaje de programación principal.
- **Hugging Face Transformers**: Para el modelo Llama 2 y el análisis de sentimientos.
- **Gradio**: Para crear una interfaz web fácil de usar.
- **Pandas**: Para gestionar el dataset de preguntas y respuestas.
- **Google Colab**: Entorno de desarrollo.
- **Torch**: Para gestionar las computaciones del modelo de IA.

---

### Guía de Instalación

1. **Clonar el Repositorio**:
   ```bash
   git clone https://github.com/usuario/asistente-salud-mental.git
   cd asistente-salud-mental
   ```

2. **Instalar las Librerías Necesarias**:
   Dentro de Google Colab o tu entorno local de Python, instala los paquetes necesarios:
   ```python
   !pip install accelerate protobuf sentencepiece torch git+https://github.com/huggingface/transformers huggingface_hub pandas gradio
   ```

3. **Configurar la Autenticación en Hugging Face**:
   Inicia sesión en Hugging Face con tu token:
   ```python
   from huggingface_hub import login
   login(token="TU_TOKEN_DE_HUGGING_FACE")
   ```

4. **Ejecutar el Chatbot**:
   Carga el notebook en Google Colab y ejecuta para iniciar la interfaz de Gradio.
   ```python
   interface.launch()
   ```

### Funcionalidad de Análisis de Sentimientos

Se utilizará el pipeline de análisis de sentimientos de Hugging Face para detectar si el usuario está en un estado emocional positivo, neutro o negativo, y ajustar las respuestas del chatbot en consecuencia.

### Estructura del Código

1. **Inicialización**:
   - Cargar el modelo Llama 2 y su tokenizador.
   - Cargar o crear el dataset de preguntas y respuestas (Q&A).

2. **Análisis de Sentimiento**:
   - Aplicar el análisis de sentimientos al input del usuario.

3. **Respuesta del Chat**:
   - Si la pregunta del usuario está en el dataset, se devuelve la respuesta correspondiente.
   - Si no, se utiliza el modelo Llama 2 para generar una respuesta.

4. **Aprendizaje Dinámico**:
   - Si el usuario hace una pregunta nueva, se agrega al dataset de preguntas y respuestas para futuras interacciones.

5. **Interfaz con Gradio**:
   - Permite a los usuarios hacer preguntas a través de una interfaz web sencilla.

---

### Capturas de Pantalla de la Versión 1.0

![captura1](imagenes/captura1.png)

![captura2](imagenes/captura2.png)

![captura3](imagenes/captura3.png)

![captura4](imagenes/captura4.png)
Lo que se observa es que al depender de el trafico de la red o version gratuita de la fuente, no responde en forma instantanea, para ello se buscara mejor con archivo csv.

### Pruebas y Retroalimentación

**Pros**:
- El análisis de sentimientos funciona para detectar emociones positivas, neutras o negativas.
- Gradio proporciona una interfaz sencilla y fácil de usar.
- El chatbot actualiza dinámicamente su dataset de preguntas y respuestas, mejorando con cada interacción.

**Áreas de Mejora**:
- **Precisión del Análisis de Sentimientos**: Las respuestas no siempre reflejan con precisión la emoción detectada.
- **Calidad de las Respuestas del Modelo**: Las respuestas generadas por Llama 2 pueden ser más empáticas y relevantes en el contexto de la salud mental.
- **Respuestas Específicas por Emoción**: Futuras versiones deberían incluir respuestas predefinidas para casos de alto estrés o depresión.
 # codigo del notebook

      !pip install accelerate protobuf sentencepiece torch git+https://github.com/huggingface/transformers huggingface_hub
      !pip install gradio
      !pip install transformers

      import os
      import pandas as pd
      from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
      from huggingface_hub import login
      import torch

      ## Autenticación en Hugging Face
      login(token="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")

      ## Cargamos el modelo de análisis de sentimientos
      sentiment_pipeline = pipeline("sentiment-analysis")

      ## Función para analizar el sentimiento
      def analyze_sentiment(text):
         result = sentiment_pipeline(text)[0]
         return result['label'], result['score']

      ## Cargar el modelo Llama 2 y el tokenizer
      model_id = "NousResearch/Llama-2-7b-chat-hf"
      model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
      tokenizer = AutoTokenizer.from_pretrained(model_id)
      tokenizer.use_default_system_prompt = False

      ## Inicializar el pipeline de generación de texto
      llama_pipeline = pipeline(
         "text-generation",
         model=model,
         tokenizer=tokenizer,
         torch_dtype=torch.float16,
         device_map="auto",
         max_length=1024
      )

      def generate_response(question):
         ## Analizamos el sentimiento de la entrada del usuario
         sentiment, score = analyze_sentiment(question)

         ## Ajustar respuesta en función del sentimiento
         if sentiment == "POSITIVE":
            response_prefix = "Me alegra escuchar eso. "
         elif sentiment == "NEGATIVE":
            response_prefix = "Lamento que te sientas así. Estoy aquí para ayudarte. "
         else:
            response_prefix = "Gracias por compartir tus pensamientos. "

         ## Generar una respuesta utilizando Llama 2
         response = llama_pipeline(question, max_length=150, do_sample=True)[0]['generated_text']

         ## Combinar el prefijo basado en sentimiento con la respuesta generada
         final_response = response_prefix + response
         return final_response


      import gradio as gr

      def gradio_chat_interface(question):
         response = generate_response(question)
         return response

      ## Crear la interfaz Gradio
      interface = gr.Interface(
         fn=gradio_chat_interface,
         inputs="text",
         outputs="text",
         title="Asistente de Salud Mental con Análisis de Sentimientos",
         description="El chatbot proporciona respuestas ajustadas según el estado emocional del usuario, ofreciendo mensajes de apoyo y ánimo."
      )

      ## Lanzar la interfaz
      interface.launch()

---

### Próximos Pasos para la Versión 2.0:
1. **Mejorar las Respuestas Basadas en Sentimientos**: Ajustar las respuestas según los estados emocionales específicos.
2. **Añadir Recursos Específicos por Emoción**: Proporcionar enlaces o recursos de apoyo emocional si se detectan emociones negativas.
3. **Añadir , Refinar el Dataset**: Incluir más preguntas y respuestas relacionadas con temas específicos de salud mental para mejorar las capacidades del chatbot.

-------------------------------------------------------------------------------------------------------------

# version 2.0
Enlace : https://colab.research.google.com/drive/18R7f5Ifojt0yZTAhBQ2Xu2vHYK8tBoqy?usp=sharing

A continuación, detallo el notebook actualizado para la **Versión 2.0** del Asistente de Salud Mental, que incluye las mejoras solicitadas: refinamiento de respuestas basadas en el sentimiento, adición de recursos específicos por emoción, y optimización del dataset con un archivo CSV para agilizar la respuesta.

### **Asistente de Salud Mental - Versión 2.0**

![captura7](imagenes/captura7.png)

#### **Instalación de las Librerías Necesarias**

```python
!pip install accelerate protobuf sentencepiece torch git+https://github.com/huggingface/transformers huggingface_hub
!pip install gradio
!pip install pandas
```

#### **Importación de Librerías**

```python
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import torch
import gradio as gr
```

#### **Autenticación en Hugging Face**

```python
# Autenticación en Hugging Face
login(token="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")
```

#### **Cargar Modelo de Análisis de Sentimientos**

```python
# Cargar el modelo de análisis de sentimientos
sentiment_pipeline = pipeline("sentiment-analysis")

# Función para analizar el sentimiento
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']
```

#### **Cargar el Modelo Llama 2 y el Tokenizer**

```python
# Cargar el modelo Llama 2 y el tokenizer
model_id = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.use_default_system_prompt = False

# Inicializar el pipeline de generación de texto
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length=1024
)
```

## Uso del Archivo CSV de Preguntas y Respuestas

El archivo CSV `preguntas_respuestas.csv` se utiliza para proporcionar respuestas predefinidas a preguntas comunes sobre salud mental en el chatbot. Si una pregunta del usuario coincide con una de las preguntas en el archivo CSV, el chatbot proporcionará la respuesta asociada. Si no hay una coincidencia, se generará una respuesta utilizando análisis de sentimientos y generación de texto mediante un modelo preentrenado.

### Formato del Archivo CSV

El archivo CSV debe seguir el siguiente formato:

```
preguntas,respuestas
```

- **preguntas**: La pregunta o inquietud del usuario. Debe ser una cadena de texto.
- **respuestas**: La respuesta predefinida asociada a la pregunta. También debe ser una cadena de texto.

### Ejemplo de Archivo CSV

A continuación se muestra un ejemplo de cómo debería estar estructurado el archivo `preguntas_respuestas.csv`:

```
preguntas,respuestas
estoy ansioso,abraza un árbol
estoy cansado,tranquilo respira profundo
estoy estresado,toma un té
estoy nervioso,siéntate y cuenta hasta 10
me siento triste,todo saldrá bien, toma tu tiempo
no puedo dormir,busca una rutina relajante antes de dormir
me siento solo,recuerda que siempre hay personas para apoyarte
me siento sin energía,haz pequeñas actividades que te gusten
tengo miedo,es normal tener miedo, poco a poco lo superarás
estoy frustrado,intenta relajarte, todo pasa con el tiempo
me siento desmotivado,encuentra algo que te apasione y empieza por pasos pequeños
me siento agobiado,tómate un descanso y da un paseo si puedes
no sé cómo manejar mis emociones,intenta escribir lo que sientes, te ayudará a organizar tus pensamientos
me siento abrumado por mis problemas,divide los problemas en pequeños pasos y resuélvelos uno a uno
siento que nadie me entiende,a veces es difícil, pero expresar cómo te sientes puede ayudarte a ser comprendido
estoy muy preocupado por el futuro,enfócate en el presente y lo que puedes controlar ahora
me siento muy solo,busca a alguien de confianza para hablar o salir a despejarte
me siento confundido,tómate un momento para reflexionar y aclarar tus ideas
no encuentro sentido a lo que hago,intenta descubrir qué te apasiona y persigue esos intereses
me siento culpable,perdónate a ti mismo, todos cometemos errores y aprendemos de ellos
me siento incapaz,recuerda que el fracaso es una parte normal del aprendizaje, sigue intentándolo
no sé qué hacer con mi vida,da pequeños pasos, no necesitas tener todo resuelto de inmediato
tengo miedo al fracaso,el fracaso es solo una oportunidad de aprender algo nuevo
siento mucha presión,tómate las cosas con calma, un paso a la vez
me siento agotado emocionalmente,busca momentos para ti y descansa tu mente
```

![captura5](imagenes/captura5.png)

### Nota

El archivo CSV es una forma eficiente de manejar respuestas predefinidas. Sin embargo, el chatbot también puede generar respuestas para preguntas no incluidas en el archivo CSV mediante el uso de modelos de generación de texto y análisis de sentimientos.



#### **Cargar Dataset de Preguntas y Respuestas (CSV)**

En esta nueva versión, el dataset de preguntas y respuestas será manejado desde un archivo CSV para optimizar el tiempo de respuesta. Asegúrate de tener un archivo `preguntas_respuestas.csv` en la misma carpeta que el notebook, con las siguientes columnas: `Pregunta`, `Respuesta`.

```python
# Cargar dataset desde un archivo CSV
def cargar_dataset():
    return pd.read_csv('preguntas_respuestas.csv')

dataset = cargar_dataset()
```

#### **Función para Buscar Respuestas en el Dataset**

Se implementa una función que busca rápidamente respuestas preexistentes en el dataset:

```python
# Función para buscar respuestas en el dataset
def buscar_respuesta(pregunta):
    if pregunta in dataset['Pregunta'].values:
        return dataset[dataset['Pregunta'] == pregunta]['Respuesta'].values[0]
    else:
        return None
```

#### **Generación de Respuestas Basadas en Sentimiento**

La función `generate_response` ahora combina la búsqueda en el dataset y el análisis de sentimiento:

```python
# Generar una respuesta basada en la entrada del usuario
def generate_response(question):
    # Buscar respuesta en el dataset
    respuesta_preexistente = buscar_respuesta(question)
    
    if respuesta_preexistente:
        return respuesta_preexistente

    # Analizar el sentimiento de la entrada del usuario
    sentiment, score = analyze_sentiment(question)

    # Ajustar respuesta en función del sentimiento
    if sentiment == "POSITIVE":
        response_prefix = "Me alegra escuchar eso. "
    elif sentiment == "NEGATIVE":
        response_prefix = "Lamento que te sientas así. Aquí tienes algunos recursos que podrían ayudarte: [Enlace a recursos]. "
    else:
        response_prefix = "Gracias por compartir tus pensamientos. "

    # Generar una respuesta utilizando Llama 2
    response = llama_pipeline(question, max_length=150, do_sample=True)[0]['generated_text']

    # Combinar el prefijo basado en sentimiento con la respuesta generada
    final_response = response_prefix + response
    
    # Agregar nueva pregunta y respuesta al dataset
    nueva_entrada = pd.DataFrame([[question, final_response]], columns=["Pregunta", "Respuesta"])
    nueva_entrada.to_csv('preguntas_respuestas.csv', mode='a', header=False, index=False)
    
    return final_response
```

#### **Interfaz Gradio Mejorada**

![captura6](imagenes/captura6.png)

```python
# Función para interfaz Gradio
def gradio_chat_interface(question):
    response = generate_response(question)
    return response

# Crear la interfaz Gradio mejorada
interface = gr.Interface(
    fn=gradio_chat_interface,
    inputs=gr.inputs.Textbox(lines=2, label="Escribe tu pregunta o inquietud"),
    outputs="text",
    title="Asistente de Salud Mental con Análisis de Sentimientos",
    description="Este chatbot ajusta sus respuestas según el estado emocional del usuario. Proporciona apoyo y recursos si se detectan emociones negativas.",
    theme="dark",  # Tema oscuro para una interfaz más moderna
    examples=[
        ["Me siento muy triste hoy."],
        ["No sé cómo manejar el estrés del trabajo."],
        ["Hoy me siento muy feliz y motivado."]
    ],
    allow_flagging=False,  # Deshabilitar el reporte de respuestas
)

# Lanzar la interfaz
interface.launch()
```

### **Resumen de Mejoras en la Versión 2.0**

- **Mejora en la Eficiencia**: Se implementa la búsqueda de preguntas en un archivo CSV para evitar la sobrecarga en el modelo, lo que reduce significativamente el tiempo de respuesta.
- **Recursos Basados en Sentimiento**: Cuando se detectan emociones negativas, se proporcionan enlaces a recursos útiles de apoyo emocional.
- **Interfaz Gradio Mejorada**: Ahora con un diseño más moderno y ejemplos predefinidos, mejorando la experiencia del usuario.

---

### **Próximos Pasos**

- **Pruebas y Optimización**: Realizar pruebas para capturar retroalimentación y hacer ajustes necesarios en el manejo del dataset.
- **Ampliar Dataset**: Continuar añadiendo preguntas y respuestas al dataset CSV para mejorar las capacidades del chatbot.
- **Interacción Multilenguaje**: Posible adición de soporte multilingüe en futuras versiones para llegar a más usuarios.

-------------------------------------------------------------------------------------------------------------

## LLegamos a una version mas de prueba, la herramienta sigue mejorando

Dado que se  quieres implementar un chatbot que maneje consultas en un "consultorio virtual" con interacción dinámica, almacenamiento de respuestas en un CSV, integración de un modelo como Llama 2 para responder a preguntas no definidas, propongo un notebook en Python con Gradio como interfaz para la interacción y Hugging Face para el análisis de sentimientos y generación de respuestas.

### Plan del Notebook

1. **Interfaz de Consultorio Virtual**:
   - Un flujo de conversación interactivo que inicia con "¿Cómo te llamas?" y sigue con una serie de preguntas basadas en la respuesta del usuario.
   - Si el chatbot no encuentra una respuesta en el CSV, utiliza el modelo de lenguaje para generar una respuesta basada en el análisis de sentimientos y otras funciones dinámicas.
   
2. **Almacenamiento en CSV**:
   - Guardar las preguntas y respuestas de las sesiones para enriquecer la base de datos de respuestas.
   - El CSV se utilizará para manejar respuestas predefinidas y personalizadas.

3. **Integración de Análisis de Sentimientos y Llama 2**:
   - Uso de análisis de sentimientos para adaptar respuestas más empáticas o adecuadas según el tono emocional del usuario.
   - Si no se encuentra la respuesta en el CSV, Llama 2 genera una respuesta adecuada basándose en el contexto.

### Estructura General

1. **Inicio del Notebook**:
   - Importación de librerías.
   - Configuración de análisis de sentimientos y modelo Llama 2.

2. **Función para manejar las preguntas y respuestas**:
   - Flujo conversacional dinámico para el consultorio.
   - Búsqueda en el CSV o generación de respuesta con el modelo de lenguaje si no se encuentra en el CSV.

3. **Almacenamiento en CSV**:
   - Registro de la conversación en un archivo CSV, con la posibilidad de reutilizar respuestas en futuras interacciones.

4. **Interfaz Gradio**:
   - Configuración de una interfaz de usuario simple y efectiva.

### Código Propuesto para el Notebook

```python
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
from huggingface_hub import login

# Autenticación en Hugging Face (añade tu propio token aquí)
login(token="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")

# Configurar análisis de sentimientos
sentiment_pipeline = pipeline("sentiment-analysis")

# Cargar modelo Llama 2
model_id = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Inicializar pipeline de generación de texto
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=150,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Cargar o crear archivo CSV
csv_file = 'consultorio_respuestas.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["pregunta", "respuesta"])

# Función para manejar el flujo de conversación
def chat_flow(user_input, conversation_state):
    if 'nombre' not in conversation_state:
        conversation_state['nombre'] = user_input
        return f"Hola {conversation_state['nombre']}, ¿qué te está pasando?", conversation_state

    if 'primera_pregunta' not in conversation_state:
        conversation_state['primera_pregunta'] = user_input
        return f"¿Por qué {user_input.lower()}?", conversation_state

    # Buscar en el CSV si hay respuesta para la pregunta
    predefined_response = find_in_csv(user_input)
    
    if predefined_response:
        return predefined_response, conversation_state

    # Si no hay respuesta en el CSV, generar una respuesta usando Llama 2
    generated_response = llama_pipeline(user_input, max_length=150)[0]['generated_text']

    # Guardar la nueva pregunta y respuesta en el CSV
    new_row = {'pregunta': user_input, 'respuesta': generated_response}
    global df
    df = df.append(new_row, ignore_index=True)
    df.to_csv(csv_file, index=False)

    return generated_response, conversation_state

# Función para buscar respuesta en el CSV
def find_in_csv(question):
    response_row = df[df['pregunta'].str.lower() == question.lower()]
    if not response_row.empty:
        return response_row.iloc[0]['respuesta']
    return None

# Función para análisis de sentimientos
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Crear interfaz con Gradio
with gr.Blocks() as interface:
    conversation_state = gr.State()
    
    with gr.Row():
        user_input = gr.Textbox(label="Escribe tu consulta")
        response_output = gr.Textbox(label="Respuesta")
    
    # Botón de envío
    submit_button = gr.Button("Enviar")
    
    # Manejar la entrada del usuario
    submit_button.click(chat_flow, inputs=[user_input, conversation_state], outputs=[response_output, conversation_state])

# Lanzar interfaz
interface.launch()
```

### Explicación de las Funciones:

1. **`chat_flow`**:
   - Este es el corazón de la conversación. Dependiendo del estado de la conversación, el chatbot responde con una nueva pregunta, consulta el CSV o genera una respuesta usando el modelo Llama 2.

2. **`find_in_csv`**:
   - Esta función busca en el archivo CSV si hay una respuesta predefinida para la pregunta del usuario.

3. **`analyze_sentiment`**:
   - Realiza un análisis de sentimientos en el input del usuario para ajustar el tono de las respuestas si es necesario.

4. **Gradio Interface**:
   - La interfaz es una aplicación sencilla donde el usuario puede escribir su consulta, y el chatbot responde dinámicamente según el estado de la conversación.

### Mejoras de la Interfaz:

1. **Flujo de Conversación Personalizado**: 
   - El chatbot pregunta el nombre del usuario y utiliza el nombre en todas las respuestas posteriores, creando una experiencia más personalizada.

2. **Almacenamiento Dinámico en CSV**: 
   - Toda conversación es almacenada en el CSV, y nuevas preguntas y respuestas enriquecen la base de datos, mejorando el rendimiento del chatbot en futuras sesiones.

3. **Sincronización con Modelos de Lenguaje**:
   - Si no se encuentra una respuesta en el CSV, el modelo Llama 2 genera una respuesta dinámica, lo que permite al chatbot manejar un amplio rango de consultas.

4. **Integración de Análisis de Sentimientos**:
   - Las respuestas pueden adaptarse en función del análisis de sentimientos, lo que hace que las interacciones sean más empáticas y personalizadas.

#### Funcionalidades:
1. **Consultorio Virtual Interactivo**:
   - El chatbot mantiene un flujo conversacional personalizado, comenzando con preguntas sobre el estado del usuario y adaptándose dinámicamente según las respuestas.
   
2. **Manejo Dinámico de Consultas**:
   - El chatbot maneja preguntas basadas en un CSV, y si no se encuentra una respuesta predefinida, se genera una respuesta utilizando el modelo Llama 2.

3. **Análisis de Sentimientos**:
   - El chatbot utiliza análisis de sentimientos para ajustar las respuestas de manera más emocionalmente apropiada.

4. **Almacenamiento de Conversaciones**:
   - Las preguntas y respuestas nuevas son almacenadas en el CSV para ser reutilizadas en futuras interacciones.

Vamos a estructurar el CSV para clasificar las preguntas y respuestas según el estado de ánimo y el grupo etario (adolescentes, adultos y tercera edad). También implementaremos el análisis de sentimiento para detectar el estado emocional del usuario y proporcionar una respuesta adecuada según la clasificación. 

### Estructura del CSV

Ampliemos el CSV con las siguientes columnas:
1. **Pregunta**: La consulta del usuario.
2. **Respuesta**: La respuesta predeterminada.
3. **Estado de ánimo**: Clasificación del estado emocional del usuario (ej. Ansioso, Triste, Cansado, etc.).
4. **Grupo etario**: Clasificación del grupo etario (Adolescentes, Adultos, Tercera edad).
5. **Recomendaciones**: Aquí podemos añadir recomendaciones adicionales (canciones, poesías, películas, etc.).

![captura8](imagenes/captura8.png)

### Ejemplo del CSV actualizado:

| Pregunta                         | Respuesta                                                   | Estado de ánimo   | Grupo etario      | Recomendaciones                                      |
|-----------------------------------|-------------------------------------------------------------|-------------------|-------------------|-----------------------------------------------------|
| Estoy ansioso                     | Abraza un árbol                                             | Ansioso           | Adolescentes      | Canción: "Nature Sounds"                             |
| Estoy ansioso                     | Haz respiración profunda y trata de desconectarte           | Ansioso           | Adultos           | Película: "La vida es bella"                        |
| Estoy ansioso                     | Busca un espacio tranquilo, escucha tu música favorita      | Ansioso           | Tercera edad      | Poesía: "Invictus" de William Ernest Henley          |
| Estoy cansado                     | Tranquilo, respira profundo                                 | Cansado           | Adolescentes      | Desafío: "Escribe 3 cosas que te motivan"           |
| Estoy cansado                     | Descansa un poco, relájate                                  | Cansado           | Adultos           | Canción: "Let it be" - The Beatles                  |
| Estoy cansado                     | Lee un libro corto o escucha música relajante               | Cansado           | Tercera edad      | Película: "Forrest Gump"                            |
| Estoy estresado                   | Toma un té y desconéctate por un momento                    | Estresado         | Adolescentes      | Canción: "Imagine" - John Lennon                    |
| Estoy estresado                   | Camina al aire libre, respira profundamente                 | Estresado         | Adultos           | Poesía: "Oda al mar" de Pablo Neruda                |
| Estoy estresado                   | Escucha sonidos de la naturaleza                            | Estresado         | Tercera edad      | Película: "Los coristas"                            |
| Me siento triste                  | Todo saldrá bien, toma tu tiempo                            | Triste            | Adolescentes      | Película: "En busca de la felicidad"                |
| No sé cómo manejar mis emociones  | Intenta escribir lo que sientes, te ayudará a organizarte   | Confundido        | Adultos           | Canción: "Fix You" - Coldplay                       |
| Tengo miedo al fracaso            | El fracaso es solo una oportunidad de aprender              | Miedoso           | Adolescentes      | Desafío: "Escribe algo que hayas superado"          |
| Me siento solo                    | Adopta una mascota                                          | Solitario         | Adolescentes      | Actividad: "Ir a un refugio de animales"            |

### Ajustes en el código para la funcionalidad:

1. **Clasificación por edad y estado de ánimo**: Al detectar el estado emocional mediante el análisis de sentimiento, se buscarán preguntas y respuestas clasificadas según el grupo etario del usuario.
2. **Apoyo con Llama 2**: Si no se encuentra una pregunta en el CSV, Llama 2 generará una respuesta dinámica.
3. **Recomendaciones personalizadas**: Si se encuentra una respuesta, se ofrecerán recomendaciones adicionales basadas en el estado de ánimo y el grupo etario.

### Actualización del código:

```python
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
from huggingface_hub import login
import threading
import random

# Autenticación en Hugging Face
login(token="tu_token")

# Configurar análisis de sentimientos
sentiment_pipeline = pipeline("sentiment-analysis")

# Cargar modelo Llama 2
model_id = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Inicializar pipeline de generación de texto
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=150,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Cargar o crear archivo CSV
csv_file = "preguntas_respuestas.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["pregunta", "respuesta", "estado_animo", "grupo_etario", "recomendaciones"])

# Función para buscar respuesta en el CSV según estado de ánimo y grupo etario
def find_in_csv(question, estado_animo, grupo_etario):
    response_row = df[(df['pregunta'].str.lower() == question.lower()) &
                      (df['estado_animo'].str.lower() == estado_animo.lower()) &
                      (df['grupo_etario'].str.lower() == grupo_etario.lower())]
    if not response_row.empty:
        return response_row.iloc[0]['respuesta'], response_row.iloc[0]['recomendaciones']
    return None, None

# Función para detener la generación después de un límite de tiempo
def llama_response_with_timeout(prompt, timeout=5):
    result = [None]  # Usamos una lista para poder modificarla dentro del hilo

    def generate():
        try:
            result[0] = llama_pipeline(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        except Exception as e:
            result[0] = f"Error generando la respuesta: {str(e)}"

    thread = threading.Thread(target=generate)
    thread.start()
    thread.join(timeout)  # Esperar hasta el tiempo límite

    if thread.is_alive():  # Si el hilo sigue corriendo, lo detenemos
        return "La respuesta está tardando demasiado, intenta de nuevo."
    return result[0]

# Función para manejar el flujo de conversación
def chat_flow(user_input, conversation_state, chat_history, grupo_etario):
    print(f"Estado inicial de la conversación: {conversation_state}")

    # Detectar el estado de ánimo mediante análisis de sentimiento
    sentiment_result = sentiment_pipeline(user_input)[0]
    estado_animo = sentiment_result['label'].lower()
    
    # Verificar si la conversación ha sido reseteada
    if conversation_state.get('reset', False):
        conversation_state.clear()
        return "La conversación ha sido reiniciada. Dime tu nombre.", conversation_state, chat_history

    if 'nombre' not in conversation_state:
        conversation_state['nombre'] = user_input
        chat_history.append(f"Tú: {user_input}")
        response = f"Hola {conversation_state['nombre']}, ¿qué te está pasando?"
        chat_history.append(f"Bot: {response}")
        return response, conversation_state, chat_history

    # Buscar en el CSV si hay respuesta para la pregunta según el estado de ánimo y el grupo etario
    predefined_response, recommendation = find_in_csv(user_input, estado_animo, grupo_etario)

    if predefined_response:
        chat_history.append(f"Tú: {user_input}")
        chat_history.append(f"Bot: {predefined_response} | Recomendación: {recommendation}")
        return f"{predefined_response} | Recomendación: {recommendation}", conversation_state, chat_history

    # Generar una respuesta usando Llama 2 con límite de tiempo si no hay respuesta en el CSV
    generated_response = llama_response_with_timeout(user_input)
    chat_history.append(f"Tú: {user_input}")
    chat_history.append(f"Bot: {generated_response}")

    # Guardar la nueva pregunta y respuesta en el CSV
    new_row = {'pregunta': user_input, 'respuesta': generated_response, 'estado_animo': estado_animo, 'grupo_etario': grupo_etario, 'recomendaciones': 'N/A'}
    global df
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_file, index=False)

    print("Respuesta generada y guardada.")
    return generated_response, conversation_state, chat_history

# Función para reiniciar la conversación
def reset_conversation():
    return "La conversación ha sido reiniciada. Dime tu nombre.", {'reset': True}, []

# Crear interfaz con Gradio
with gr.Blocks() as interface:
    conversation_state = gr.State(value={})  # Iniciar el estado de la conversación como un diccionario vacío
    chat_history = gr.State(value=[])  # Historial de conversación
    grupo_etario = gr.Dropdown(choices=["Adolescentes", "Adultos", "Tercera edad"], label="Grupo Etario")
    
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image("perfil.jpg", label="Perfil", interactive=False)  # Imagen de perfil
        with gr.Column(scale=2):


            chatbot = gr.Chatbot()  # Ventana de chat
            user_input = gr.Textbox(label="Tu mensaje")
            submit_button = gr.Button("Enviar")
            reset_button = gr.Button("Resetear")

    # Vincular funciones con los botones
    submit_button.click(chat_flow, inputs=[user_input, conversation_state, chat_history, grupo_etario], outputs=[chatbot, conversation_state, chat_history])
    reset_button.click(reset_conversation, outputs=[chatbot, conversation_state, chat_history])

# Ejecutar la interfaz
interface.launch()
```

### Siguientes pasos:
1. Multiplicar las preguntas y respuestas por grupo etario y estado de ánimo en el CSV.
2. Implementar más recomendaciones dinámicas según el estado de ánimo.
3. Realizar pruebas de la función de generación con Llama 2.


# Ultima version hasta el momento


### **- Chatbot para Consultorio Virtual con Análisis de Sentimientos y Modelo meta-llama/Llama-2-7b-chat-hf**



#### **Descripción del Proyecto**

Este proyecto es un **chatbot de consultorio virtual** diseñado para ofrecer interacciones personalizadas y relevantes utilizando un modelo de lenguaje avanzado (**meta-llama/Llama-2-7b-chat-hf** de Hugging Face) y un **análisis de sentimientos** que ajusta las respuestas del chatbot según el estado emocional del usuario. 

El chatbot permite a los usuarios interactuar en un entorno relajante con música y videos motivacionales, mientras responde a preguntas de manera fluida y adaptada al estado emocional. El administrador tiene la capacidad de cargar música y videos y reentrenar el modelo cuando sea necesario. El proyecto está implementado en **Google Colab**, utilizando **Python** y **Hugging Face** como núcleo de su funcionalidad.

![captura9](imagenes/captura9.png)

#### **Características Principales**

1. **Análisis de Sentimientos**:
   - Se utiliza un modelo de análisis de sentimientos de **Hugging Face** que detecta el estado emocional del usuario (positivo, negativo, neutro).
   - El chatbot ajusta su tono y contenido de respuesta según el resultado del análisis de sentimientos, generando respuestas empáticas, motivacionales, o neutrales.

2. **Generación de Texto con Modelo meta-llama/Llama-2-7b-chat-hf**:
   - El modelo **meta-llama/Llama-2-7b-chat-hf** genera respuestas coherentes y detalladas a las preguntas del usuario.
   - El modelo puede mantener una conversación fluida, proporcionando información útil y relevante.

3. **Sección de Pacientes**:
   - Los pacientes pueden interactuar con el chatbot, escuchar **música relajante**, y ver **videos motivacionales** proporcionados por el administrador.
   - La experiencia se adapta según el estado emocional detectado para mejorar la interacción y ofrecer una mejor atención.

4. **Sección de Administrador**:
   - El administrador puede cargar **música** y **videos** para el consultorio, permitiendo que los pacientes accedan a estos recursos.
   - Funcionalidad para **reentrenar** el chatbot si es necesario para mejorar la calidad de las respuestas.

![captura10](imagenes/captura10.png)

#### **Requisitos del Proyecto**

1. **Análisis de Sentimientos**:
   - El proyecto integra un análisis de sentimientos de Hugging Face para ajustar el tono y el contenido de las respuestas del chatbot.
   - Dependiendo del estado emocional detectado, las respuestas pueden ser motivacionales (estado negativo), alentadoras (estado positivo) o neutrales (otros estados).

2. **Modelo de Generación de Texto**:
   - El proyecto utiliza el modelo **meta-llama/Llama-2-7b-chat-hf** de Hugging Face, que responde a las preguntas del usuario de manera detallada y educada.
   - El modelo es responsable de generar todo el contenido de las respuestas del chatbot, sin depender de un archivo CSV o almacenamiento de respuestas.

3. **Interactividad Mejorada**:
   - El chatbot ajusta las respuestas no solo en contenido, sino también en tono, dependiendo de los resultados del análisis de sentimientos.
   - La **sección de Pacientes** ofrece una experiencia inmersiva con música y videos que ayudan a crear un ambiente relajante para los usuarios.

![captura12](imagenes/captura12.png)

#### **Estructura del Proyecto**

- **Análisis de Sentimientos**: Se utiliza el pipeline de Hugging Face para detectar el estado emocional del usuario a partir de su entrada de texto.
  
- **Modelo de Lenguaje (meta-llama/Llama-2-7b-chat-hf)**: 
   - Genera respuestas según el input del usuario y el análisis de sentimientos.
   - Mantiene un flujo de conversación fluido y detallado.

- **Sección de Pacientes**:
   - Incluye un chatbot interactivo que responde de acuerdo al estado emocional detectado.
   - Proporciona música relajante y videos motivacionales cargados por el administrador.

- **Sección de Administrador**:
   - Permite al administrador cargar música y videos para los pacientes.
   - Ofrece la funcionalidad de reentrenar el modelo cuando sea necesario.

![captura11](imagenes/captura11.png)

#### **Ejecución del Proyecto**

1. **Instalación de Dependencias**:
   - Ejecuta los siguientes comandos para instalar las bibliotecas necesarias:
   ```bash
   !pip install accelerate protobuf sentencepiece torch git+https://github.com/huggingface/transformers huggingface_hub
   !pip install gradio
   ```

2. **Configuración del Modelo**:
   - El modelo de lenguaje y el análisis de sentimientos se configuran utilizando **Hugging Face**. Debes autenticarte con un token de Hugging Face para acceder al modelo meta-llama/Llama-2-7b-chat-hf.
   ```python
   from huggingface_hub import login
   login(token="tu_token_aqui")
   ```

3. **Ejecutar la Aplicación**:
   - Para ejecutar la aplicación completa, simplemente inicia la interfaz con el siguiente comando:
   ```python
   main_app().launch()
   ```

---

#### **Detalles Técnicos Importantes**

1. **Análisis de Sentimientos**:
   - Se utiliza un pipeline de **Hugging Face** para analizar el sentimiento de la entrada del usuario, categorizándola en "positivo", "negativo", o "neutral".

2. **Generación de Respuestas**:
   - Dependiendo del análisis de sentimientos, el chatbot ajusta el **tono** y el **contenido** de las respuestas utilizando el modelo meta-llama/Llama-2-7b-chat-hf.
   - Respuestas más empáticas cuando el usuario está negativo, respuestas alegres si el usuario es positivo, y respuestas neutrales cuando el estado emocional es desconocido o neutral.

3. **Cargar Música y Videos**:
   - El administrador puede cargar música y videos, los cuales serán reproducidos en la interfaz de paciente.

4. **Reentrenar el Chatbot**:
   - Si es necesario, el administrador puede ejecutar el comando de **reentrenamiento** para ajustar el comportamiento del chatbot, aunque no depende de un archivo CSV.

---

#### **Funcionalidades del Chatbot**

- **Respuestas basadas en el análisis de sentimientos**:
  - El chatbot ajusta el tono y contenido de las respuestas dependiendo de la emoción detectada en la entrada del usuario.
  
- **Música relajante y videos motivacionales**:
  - Los pacientes pueden acceder a música relajante y videos cargados por el administrador, ayudando a crear una experiencia relajante en el consultorio virtual.

- **Panel de Administrador**:
  - El administrador puede gestionar el contenido multimedia y ajustar el chatbot.

---

#### **Contribuciones Futuras**

1. **Mejora del análisis de sentimientos**:
   - Se podrían integrar modelos de análisis de emociones más complejos, para identificar sentimientos más específicos.

2. **Entrenamiento adicional del modelo**:
   - A medida que se recopilan más interacciones, se puede considerar entrenar el modelo de lenguaje con datos adicionales para mejorar su capacidad de responder a situaciones específicas del consultorio virtual.

---

#### **Conclusión**

Este proyecto demuestra una **solución de chatbot personalizada para un consultorio virtual**. Combina el poder de los modelos de lenguaje de **Hugging Face** con un **análisis de sentimientos** que adapta las respuestas del bot, haciendo que la experiencia del usuario sea más interactiva y empática. Además, la capacidad de cargar **música** y **videos motivacionales** mejora el ambiente del consultorio, haciendo que el paciente se sienta más relajado y atendido.

Enlace : https://colab.research.google.com/drive/1OH1cqe0rjH0FuncGFJia6zL_sxE0TXFg?usp=sharing

![captura13](imagenes/captura13.png)


### Cómo Contribuir

Siéntete libre de hacer un fork del repositorio, enviar pull requests y reportar problemas en la sección de issues de GitHub.


### Contacto

Para cualquier duda o sugerencia, contacta a `claudiodrllo2023@gmail.com`.

