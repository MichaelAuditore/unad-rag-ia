# Usa una imagen base liviana
FROM python:3.10-slim

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia primero los requerimientos (para aprovechar la caché de Docker)
COPY requirements.txt .

# Instala dependencias necesarias del sistema (curl, fonts, etc.)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Exponer el puerto donde correrá Streamlit
EXPOSE 7860

# Comando principal que ejecutará la app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
