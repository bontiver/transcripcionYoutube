import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from gensim.models import LdaModel
from gensim import corpora
import nltk
from nltk.corpus import stopwords

# Descargar recursos necesarios para nltk (si es la primera vez que se ejecuta)
nltk.download('punkt')
nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('spanish')) | set(stopwords.words('english')) | {'solo'}

# Función para obtener la transcripción con tiempos
@st.cache_data(show_spinner=False)
def obtener_transcripcion_con_tiempos(video_id, idiomas=['es', 'en']):
    try:
        transcripciones = YouTubeTranscriptApi.list_transcripts(video_id)
        for idioma in idiomas:
            try:
                transcripcion = transcripciones.find_transcript([idioma])
                transcripcion_datos = transcripcion.fetch()
                
                transcripcion_con_tiempos = []
                for item in transcripcion_datos:
                    tiempo = item['start']
                    texto = item['text']
                    transcripcion_con_tiempos.append({'Tiempo (segundos)': tiempo, 'Texto': texto})
                
                texto_transcripcion = " ".join([item['text'] for item in transcripcion_datos])
                return texto_transcripcion, transcripcion_con_tiempos, idioma
            except NoTranscriptFound:
                continue
        st.warning(f"No se encontró transcripción en los idiomas seleccionados: {', '.join(idiomas)}.")
        return None, None, None
    except TranscriptsDisabled:
        st.error("Las transcripciones están deshabilitadas para este video.")
        return None, None, None
    except NoTranscriptFound:
        st.error("No se encontró ninguna transcripción para este video.")
        return None, None, None
    except Exception as e:
        st.error(f"Error inesperado al obtener la transcripción: {str(e)}")
        return None, None, None

# Función para extraer el ID del video desde la URL de YouTube
def extraer_id_video(url):
    patron = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/|.*?v=)|youtu\.be\/)([\w\-]{11})'
    match = re.match(patron, url)
    if match:
        return match.group(1)
    else:
        return None

# Función para resaltar palabras clave en la transcripción
def resaltar_palabras_clave(transcripcion, palabras_clave):
    if palabras_clave:
        for palabra in palabras_clave:
            transcripcion = transcripcion.replace(palabra, f"**{palabra}**")
    return transcripcion

# Función para análisis básico del contenido
def analizar_transcripcion(texto):
    palabras = texto.lower().split()
    palabras_filtradas = [palabra for palabra in palabras if palabra not in STOP_WORDS]
    cantidad_palabras = len(palabras_filtradas)
    palabras_comunes = Counter(palabras_filtradas).most_common(10)
    return cantidad_palabras, palabras_comunes

# Función para generar una nube de palabras
def generar_nube_palabras(texto):
    palabras_filtradas = [palabra for palabra in texto.lower().split() if palabra not in STOP_WORDS]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(palabras_filtradas))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Función para analizar el sentimiento
def analizar_sentimiento(texto):
    blob = TextBlob(texto)
    sentimiento = blob.sentiment.polarity
    return sentimiento

# Función para detectar temas usando LDA
def detectar_temas(texto, num_temas=3):
    palabras = [palabra for palabra in texto.lower().split() if palabra not in STOP_WORDS]
    diccionario = corpora.Dictionary([palabras])
    corpus = [diccionario.doc2bow(palabras)]
    
    # LDA para detectar los temas
    lda_model = LdaModel(corpus, num_topics=num_temas, id2word=diccionario, passes=15)
    temas = lda_model.print_topics(num_words=5)
    return temas

# Función modificada para generar enlaces con tiempos convertidos a minutos, agrupando cada 10 segundos
def generar_enlaces_tiempos_agrupados(transcripcion_con_tiempos, video_id):
    # Inicializamos variables
    transcripcion_agrupada = []
    texto_acumulado = ""
    tiempo_inicial = None

    for item in transcripcion_con_tiempos:
        # Redondear el tiempo al múltiplo de 10 segundos más cercano
        tiempo_en_segundos = item['Tiempo (segundos)']
        tiempo_redondeado = round(tiempo_en_segundos / 10) * 10
        
        if tiempo_inicial is None:
            # Inicializamos el tiempo
            tiempo_inicial = tiempo_redondeado
        
        if tiempo_redondeado == tiempo_inicial:
            # Acumulamos el texto para el mismo intervalo de tiempo
            texto_acumulado += " " + item['Texto']
        else:
            # Añadimos la transcripción acumulada al grupo anterior
            tiempo_en_minutos = tiempo_inicial / 60
            minutos = int(tiempo_en_minutos)
            segundos = int((tiempo_en_minutos - minutos) * 60)
            tiempo_formateado = f"{minutos}:{segundos:02d}"  # Formato mm:ss
            
            url = f"https://www.youtube.com/watch?v={video_id}&t={int(tiempo_inicial)}s"
            transcripcion_agrupada.append({'Tiempo': tiempo_formateado, 'Texto': texto_acumulado.strip(), 'URL': url})

            # Reiniciamos las variables para el nuevo grupo
            tiempo_inicial = tiempo_redondeado
            texto_acumulado = item['Texto']

    # No olvidar agregar el último grupo al final
    if texto_acumulado:
        tiempo_en_minutos = tiempo_inicial / 60
        minutos = int(tiempo_en_minutos)
        segundos = int((tiempo_en_minutos - minutos) * 60)
        tiempo_formateado = f"{minutos}:{segundos:02d}"

        url = f"https://www.youtube.com/watch?v={video_id}&t={int(tiempo_inicial)}s"
        transcripcion_agrupada.append({'Tiempo': tiempo_formateado, 'Texto': texto_acumulado.strip(), 'URL': url})

    # Mostrar la transcripción agrupada en Streamlit
    for item in transcripcion_agrupada:
        st.markdown(f"[{item['Tiempo']}] [{item['Texto']}]({item['URL']})")

# Título de la aplicación en Streamlit
st.title("Transcriptor de Videos de YouTube")

# Entrada de usuario para la URL del video
video_url = st.text_input("Introduce la URL completa del video de YouTube", "")

# Si hay una URL válida, extraer el ID y mostrar vista previa del video
video_id = extraer_id_video(video_url)
if video_id:
    st.video(f"https://www.youtube.com/watch?v={video_id}")
else:
    st.warning("Introduce una URL válida de YouTube para ver la vista previa del video.")

# Selección de idioma
idiomas_seleccionados = st.multiselect("Selecciona los idiomas de la transcripción (prioridad de búsqueda)", 
                                      ["es", "en", "fr", "de"], default=["es", "en"])

# Opción para mostrar la transcripción con tiempos
mostrar_con_tiempos = st.checkbox("Mostrar transcripción con tiempos", value=False)

# Entrada para palabras clave que se resaltarán en la transcripción
palabras_clave_input = st.text_input("Introduce palabras o frases clave para resaltar (separadas por comas)", "")

# Procesar las palabras clave introducidas
palabras_clave = [palabra.strip() for palabra in palabras_clave_input.split(",") if palabra.strip()]

# Botón para obtener la transcripción
if st.button("Obtener Transcripción"):
    if video_id:
        with st.spinner('Procesando...'):
            texto_transcripcion, transcripcion_con_tiempos, idioma_usado = obtener_transcripcion_con_tiempos(video_id, idiomas_seleccionados)

            if texto_transcripcion:
                st.subheader(f"Transcripción completa (Idioma: {idioma_usado.upper()})")
                
                # Resaltar las palabras clave en la transcripción
                texto_resaltado = resaltar_palabras_clave(texto_transcripcion, palabras_clave)

                # Mostrar la transcripción con o sin tiempos
                if mostrar_con_tiempos:
                    st.write("Transcripción interactiva agrupada cada 10 segundos:")
                    generar_enlaces_tiempos_agrupados(transcripcion_con_tiempos, video_id)
                else:
                    with st.expander("Ver transcripción completa"):
                        st.markdown(texto_resaltado)
                
                st.download_button(label="Descargar Transcripción Completa", 
                                   data=texto_transcripcion, 
                                   file_name="transcripcion.txt", 
                                   mime="text/plain")
                
                if mostrar_con_tiempos:
                    transcripcion_con_tiempos_txt = "\n".join([f"[{item['Tiempo (segundos)']:.2f}] {item['Texto']}" for item in transcripcion_con_tiempos])
                    st.download_button(label="Descargar Transcripción con Tiempos", 
                                       data=transcripcion_con_tiempos_txt, 
                                       file_name="transcripcion_con_tiempos.txt", 
                                       mime="text/plain")

                # Análisis de contenido de la transcripción
                st.subheader("Análisis de contenido")
                total_palabras, palabras_comunes = analizar_transcripcion(texto_transcripcion)
                st.write(f"**Total de palabras en la transcripción:** {total_palabras}")
                st.write("**Palabras más comunes:**")
                for palabra, frecuencia in palabras_comunes:
                    st.write(f"- {palabra}: {frecuencia} veces")

                # Análisis de sentimiento
                sentimiento = analizar_sentimiento(texto_transcripcion)
                st.write(f"**Análisis de sentimiento:** {'Positivo' if sentimiento > 0 else 'Negativo' if sentimiento < 0 else 'Neutral'}")

                # Detectar temas
                st.subheader("Temas principales del video")
                temas = detectar_temas(texto_transcripcion)
                for idx, tema in temas:
                    st.write(f"Tema {idx + 1}: {tema}")
                
                # Nube de palabras
                st.subheader("Nube de palabras")
                generar_nube_palabras(texto_transcripcion)

    else:
        st.error("La URL del video de YouTube no es válida. Asegúrate de ingresar una URL correcta.")
