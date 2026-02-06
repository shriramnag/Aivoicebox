import gradio as gr
from TTS.api import TTS
import torch
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# 1. मॉडल सेटअप
device = "cuda" if torch.cuda.is_available() else "cpu"
# लाइसेंस एग्रीमेंट के लिए एनवायरनमेंट वेरिएबल
os.environ["COQUI_TOS_AGREED"] = "1"

print("🚀 मॉडल लोड हो रहा है...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, remove_silence):
    output_path = "output.wav"
    
    # वॉयस क्लोनिंग
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_sample,
        language="hi",
        file_path=output_path,
        speed=speed,
        pitch=pitch
    )
    
    # साइलेंस रिमूवर (Silence Remover Button)
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        output_path = "clean_final.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# 2. इंटरफ़ेस (UI) - एरर से बचने के लिए सबसे सरल तरीका
# 'theme' और 'dark_mode' के झंझट को खत्म किया गया है
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ **एआई वॉयस बॉक्स - फाइनल फिक्स**")
    
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="हिंदी टेक्स्ट यहाँ लिखें")
            audio_ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath")
            
            with gr.Row():
                speed_slider = gr.Slider(0.5, 2.0, value=1.0, label="Speed (गति)")
                pitch_slider = gr.Slider(-10, 10, value=0, label="Pitch (पिच)")
            
            silence_check = gr.Checkbox(label="Silence Remover", value=True)
            submit_btn = gr.Button("🚀 आवाज़ बनाएँ", variant="primary")
        
        with gr.Column():
            audio_out = gr.Audio(label="आउटपुट")

    submit_btn.click(
        fn=generate_voice, 
        inputs=[txt, audio_ref, speed_slider, pitch_slider, silence_check], 
        outputs=audio_out
    )

if __name__ == "__main__":
    # डार्क मोड अब यहाँ से कंट्रोल होगा
    demo.launch(share=True, dark_mode=True)
