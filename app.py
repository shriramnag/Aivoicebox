import gradio as gr # या जो भी UI आप इस्तेमाल कर रहे हैं
from tts_engine import generate_voice
from brain import save_to_memory

def shriram_vani_ui(text):
    if not text.strip():
        return "कृपया कुछ लिखें..."

    try:
        # 1. वॉयस जनरेट करें (यह अंदर ही अंदर brain.py का उपयोग करेगा)
        # इसमें आपकी 0.9 Deep Match और 1.0 Emotion सेटिंग्स लॉक हैं
        output_path = generate_voice(text, output_file="shriram_output.wav")
        
        # 2. प्रोग्रेस दिखाएँ
        return output_path
    
    except Exception as e:
        return f"त्रुटि: {str(e)}"

# इंटरफ़ेस सेटअप (उदाहरण के लिए)
interface = gr.Interface(
    fn=shriram_vani_ui,
    inputs=gr.Textbox(lines=5, placeholder="यहाँ अपनी स्क्रिप्ट लिखें..."),
    outputs=gr.Audio(type="filepath"),
    title="🚩 श्रीराम वाणी - AI वॉइस मास्टर 🚩",
    description="टर्बो हाई स्पीड और 1000% मानवीय अहसास के साथ।"
)

if __name__ == "__main__":
    interface.launch()
