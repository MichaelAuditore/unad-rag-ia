import gradio as gr
import argparse
from rag_agent import build_rag

def chat_with_ai(message, history):
    response = rag_agent.invoke({"question": message})
    answer = response["result"]
    yield answer  # Streaming de respuesta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true", help="Reconstruye la base de conocimiento.")
    args = parser.parse_args()

    rag_agent = build_rag(reindex=args.reindex)

    chatbot = gr.ChatInterface(
        fn=chat_with_ai,
        title="🎓 Asistente UNAD IA",
        description="Consulta programas académicos, políticas y reglamentos de la UNAD.",
        theme="soft",
        type="messages",
        streaming=True
    )

    chatbot.launch(server_name="0.0.0.0", share=False)
