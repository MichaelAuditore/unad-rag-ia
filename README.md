**UNAD RAG AI – Asistente Académico Inteligente con IA Generativa Local**

Este proyecto implementa un **prototipo funcional (TRL 5–6)** de un asistente inteligente para la **Universidad Nacional Abierta y a Distancia (UNAD)**, desarrollado con tecnologías **de código abierto** y ejecutado **localmente**, sin dependencia de servicios en la nube.

El sistema integra un modelo **RAG (Retrieval-Augmented Generation)** con **memoria conversacional** y una **interfaz web interactiva**, capaz de responder preguntas sobre **programas académicos, políticas de gratuidad y reglamentos institucionales**.

El agente está construido sobre el ecosistema **LangChain + Ollama + ChromaDB**, empleando modelos locales como **Llama 3** o **Mistral**, y se ejecuta completamente en entorno **Python** mediante **Gradio** como interfaz gráfica.

🔹 **Características principales:**

* Recuperación inteligente de información institucional (RAG local).
* Memoria conversacional para mantener el contexto del diálogo.
* Integración con documentos PDF, TXT y fuentes web de la UNAD.
* Arquitectura modular y extensible.
* Código abierto y ejecutable sin conexión a Internet.

🔹 **Tecnologías utilizadas:**

* 🐍 Python 3.10+
* 🧩 LangChain
* 🤖 Ollama (modelos Llama 3 / Mistral)
* 🧠 ChromaDB
* 💬 Gradio (interfaz)
* 🧾 Sentence Transformers

---

## ⚙️ Requisitos previos

Antes de comenzar, asegúrate de tener instalado:

* 🐳 **Docker** y **Docker Compose**
* 💾 Al menos **6–8 GB de RAM**
* 📁 Espacio libre de **5–10 GB** (según el modelo elegido)

---

## 🚀 Instalación y ejecución

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/tuusuario/unad-rag-agent
cd unad-rag-agent
```

### 2️⃣ Agregar documentos de conocimiento

Coloca tus archivos `.pdf` o `.txt` dentro del directorio `knowledge/`.
Por ejemplo:

```
knowledge/
├── programas_academicos.pdf
├── reglamento_estudiantil.pdf
└── politicas_institucionales.txt
```

### 3️⃣ Configurar variables de entorno

Copia el archivo `.env.example` y renómbralo a `.env`:

```bash
cp .env.example .env
```

Puedes ajustar el modelo a utilizar:

```bash
OLLAMA_MODEL=phi3:3.8b
```

### 4️⃣ Iniciar el asistente

Levanta todo el sistema (Ollama + App):

```bash
docker compose up --build
```

Una vez iniciado, abre tu navegador en:
👉 **[http://localhost:7860](http://localhost:7860)**

---

## 🧠 Cambiar modelo LLM

Puedes modificar el modelo en `.env` o en `docker-compose.yml`.
Los más recomendados son:

| Modelo       | Tamaño  | Características                                      |
| ------------ | ------- | ---------------------------------------------------- |
| `mistral:7b`   | ~2 GB   | Muy rápido, eficiente, ideal para respuestas simples |
| `phi3:3.8b`  | ~3.8 GB | Excelente en español y razonamiento                  |
| `mistral:7b` | ~7 GB   | Más potente, buena comprensión contextual            |
| `llama3:8b`  | ~8 GB   | Buen equilibrio entre velocidad y calidad            |

Después de cambiar el modelo, simplemente ejecuta:

```bash
docker compose restart unad_rag
```

---

## 🔄 Reconstruir la base de conocimiento

Si agregas nuevos documentos o deseas regenerar el índice:

```bash
docker compose run unad_rag python app.py --reindex
```

Esto recreará la base vectorial (`db/chroma`).

---

## 🧩 Arquitectura interna

* **Gradio** → Interfaz de chat web
* **LangChain** → Orquestación RAG y memoria conversacional
* **Chroma** → Almacenamiento vectorial
* **Ollama** → Motor local de modelos open-source
* **SentenceTransformers** → Generación de embeddings

---

## 🧪 Ejecutar localmente sin Docker

Si prefieres ejecutar directamente en tu máquina:

```bash
pip install -r requirements.txt
python app.py --reindex
```

Y accede desde: [http://localhost:7860](http://localhost:7860)

---

## 🧰 Comandos útiles

| Acción                    | Comando                                               |
| ------------------------- | ----------------------------------------------------- |
| Levantar todo             | `docker compose up --build`                           |
| Reconstruir base de datos | `docker compose run unad_rag python app.py --reindex` |
| Instalar modelo LLM       | `docker exec -it ollama ollama pull mistral:7b` |
| Instalar modelo embedding | `docker exec -it ollama ollama pull nomic-embed-text` |
| Cambiar modelo            | Edita `.env` y ejecuta `docker compose restart`       |
| Detener contenedores      | `docker compose down`                                 |

---

## 💡 Consejos

* Guarda tus documentos organizados en `knowledge/`.
* Usa modelos pequeños si tu equipo tiene poca RAM.
* No necesitas conexión a Internet después de descargar el modelo.

---

## 🧾 Licencia

Este proyecto es **open-source** y puede modificarse libremente con fines educativos y de investigación.

---

📍 **Nivel de madurez tecnológica:** TRL 5–6 (validación de sistema completo en entorno relevante).

📂 **Licencia:** MIT

👤 **Autores:** Miguel Ángel Parada Cañon, Tania Parrado Rojas

🏫 **Escuela:** ECBTI – Ingeniería de Sistemas – UNAD
