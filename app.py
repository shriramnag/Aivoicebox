import os
import gradio as gr
from TTS.api import TTS

# 🚩 मॉडल लोड करना (बिना किसी एरर के सीधे आपके फोल्डर से) [cite: 2026-02-16]
print("🚀 श्रीराम वाणी मॉडल लोड हो रहा है...")

# आपके गिटहब/कोलाब फोल्डर का पाथ
MODEL_PATH = "/content/shriram-voice-box/Ramai.pth" 
CONFIG_PATH = "/content/shriram-voice-box/config.json"
SPEAKER_WAV = "/content/shriram-voice-box/speaker.wav"

# XTTS v2 लोड करना
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

def generate_shriram_voice(input_text):
    if not input_text.strip():
        return "कृपया कुछ टेक्स्ट लिखें..."
    
    output_file = "shriram_final_output.wav"
    
    try:
        # 🎙️ आपकी मास्टर सेटिंग्स (Locked)
        tts.tts_to_file(
            text=input_text,
            speaker_wav=SPEAKER_WAV, 
            language="hi",
            file_path=output_file,
            speed=1.0,           # टर्बो हाई स्पीड [cite: 2026-01-06]
            repetition_penalty=10.0, # हकलाहट रोकने के लिए
            temperature=0.75     # इमोशन और गहराई के लिए
        )
        return output_file
    
    except Exception as e:
        return f"त्रुटि: {str(e)}"

# 🚩 ग्राफिकल इंटरफेस (UI)
with gr.Blocks(title="🚩 श्रीराम वाणी - AI मास्टर 🚩") as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - हिंदी वॉयस क्लोन टर्बो")
    gr.Markdown("### आपकी पुरानी वर्किंग सेटिंग्स के साथ [cite: 2026-02-16]")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="अपनी स्क्रिप्ट यहाँ लिखें", 
                lines=8, 
                placeholder="यहाँ हिंदी टेक्स्ट पेस्ट करें..."
            )
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="सुनिए श्रीराम वाणी", type="filepath")

    # बटन क्लिक एक्शन
    btn.click(fn=generate_shriram_voice, inputs=input_box, outputs=output_audio)

# 🚩 पब्लिक यूआरएल (Public URL) के लिए शेयर चालू करना
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
