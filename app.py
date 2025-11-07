import gradio as gr
from rag_agent import rag_agent

def chat_with_ai(message, history):
    response = rag_agent.invoke(message)
    answer = response
    return answer

chatbot = gr.ChatInterface(
    fn=chat_with_ai,
    title="Asistente UNAD IA",
    description="Consulta programas académicos, políticas y reglamentos de la UNAD.",
    theme="soft",
    type="messages"    
)

if __name__ == "__main__":
    chatbot.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
