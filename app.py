import os
import gradio as gr
from tts_engine import generate_voice # यह आपके हगिंग फेस मॉडल को चलाएगा
from brain import save_to_memory

# 🚩 आपकी लॉक की हुई मुख्य सेटिंग्स
def generate_shriram_audio(input_text):
    if not input_text.strip():
        return "कृपया कुछ टेक्स्ट लिखें..."
    
    try:
        # यहाँ 'generate_voice' के अंदर आपकी 0.9 Deep Match 
        # और 1.0 Emotion सेटिंग्स को फिक्स रखा गया है।
        output_file = generate_voice(input_text)
        
        # याददाश्त में सेव करना ताकि मॉडल भविष्य में खुद सीखे [cite: 2026-02-16]
        save_to_memory(input_text)
        
        return output_file
    
    except Exception as e:
        return f"त्रुटि: {str(e)}"

# इंटरफ़ेस (UI) - इसमें कोई बदलाव नहीं है
with gr.Blocks(title="🚩 श्रीराम वाणी - AI मास्टर 🚩") as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - AI वॉयस क्लोनिंग")
    gr.Markdown("### टर्बो हाई स्पीड और शुद्ध उच्चारण [cite: 2026-01-06]")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=10)
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="सुनिए श्रीराम वाणी", type="filepath")

    btn.click(fn=generate_shriram_audio, inputs=input_box, outputs=output_audio)

if __name__ == "__main__":
    demo.launch()
