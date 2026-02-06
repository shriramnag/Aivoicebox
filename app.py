import gradio as gr
from TTS.api import TTS
import torch
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# --- मॉडल सेटअप (Turbo GPU/CPU) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# कोलायब में 'TTS' की जगह 'coqui-tts' का इस्तेमाल हो रहा है
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, remove_silence):
    output_path = "output.wav"
    
    # 1. वॉयस क्लोनिंग (Strict Hindi Mode)
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_sample,
        language="hi",
        file_path=output_path,
        speed=speed,
        pitch=pitch
    )
    
    # 2. साइलेंस रिमूवर (Silence Remover Button Logic)
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        output_path = "clean_final.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# --- इंटरफ़ेस (Updated Gradio UI) ---
# 'dark_mode' एरर को हटाने के लिए नई थीम सेटिंग्स
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange", secondary_hue="gray")) as demo:
    # डार्क मोड को जबरदस्ती लागू करने के लिए जावास्क्रिप्ट
    demo.load(None, None, None, _js="() => { document.body.classList.add('dark'); }")
    
    gr.Markdown("# 🎙️ **एआई वॉयस बॉक्स - टर्बो अपडेट**")
    gr.Markdown("हिंदी वॉयस क्लोनिंग, पिच और स्पीड कंट्रोल के साथ।")

    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="हिंदी टेक्स्ट यहाँ लिखें", placeholder="जैसे: जय श्री गणेश, कैसे हैं आप?")
            audio_ref = gr.Audio(label="अपना .wav वॉयस सैंपल दें", type="filepath")
            
            with gr.Row():
                speed_slider = gr.Slider(0.5, 2.0, value=1.0, label="गति (Speed)")
                pitch_slider = gr.Slider(-10, 10, value=0, label="पिच (Pitch)")
            
            silence_check = gr.Checkbox(label="फालतू सन्नाटा हटाएँ (Silence Remover)", value=True)
            submit_btn = gr.Button("🚀 आवाज़ बनाएँ (Generate)", variant="primary")
        
        with gr.Column():
            audio_out = gr.Audio(label="आपका फाइनल ऑडियो")

    submit_btn.click(
        fn=generate_voice, 
        inputs=[txt, audio_ref, speed_slider, pitch_slider, silence_check], 
        outputs=audio_out
    )

# लॉन्च सेटिंग्स
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
