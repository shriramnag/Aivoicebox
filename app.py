import os
import torch
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import split_on_silence
import re

# टर्बो सेटअप (2026 अपडेट)
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def perfect_hindi_cleaner(text):
    # यह सिर्फ हिंदी अक्षरों (अ-ज्ञ) और पूर्ण विराम (।) को रहने देगा
    # बाकी सब कुछ (चीनी/अंग्रेजी अक्षर) अपने आप साफ हो जाएगा
    clean_text = re.sub(r'[^\u0900-\u097F\s।,.?]', '', text)
    return clean_text

def generate_voice(text, voice_sample, remove_silence):
    # 1. टेक्स्ट को शुद्ध करना
    clean_text = perfect_hindi_cleaner(text)
    output_path = "final_shriram_voice.wav"
    
    # 2. वॉयस जनरेशन (Strict Mode)
    # split_sentences=True हकलाने को रोकता है
    tts.tts_to_file(
        text=clean_text, 
        speaker_wav=voice_sample, 
        language="hi",              # हिंदी भाषा पर पूर्ण नियंत्रण
        file_path=output_path,
        split_sentences=True        
    )
    
    # 3. साइलेंस रिमूवर (बिना किसी देरी के)
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        output_path = "clean_turbo_output.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# --- इंटरफ़ेस (Dark Mode) ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    # एरर फ्री डार्क मोड
    demo.load(None, None, None, _js="() => { document.body.classList.add('dark'); }")
    gr.Markdown("# 🎙️ **श्रीराम वाणी - शुद्ध हिंदी इंजन (v2)**")
    
    with gr.Row():
        with gr.Column():
            txt_input = gr.Textbox(label="सिर्फ हिंदी लिखें", value="नमस्ते, अब मैं सिर्फ शुद्ध हिंदी बोलूँगा।")
            audio_ref = gr.Audio(label="वॉइस सैंपल (.wav)", type="filepath")
            silence_on = gr.Checkbox(label="सन्नाटा हटाएँ (Silence Remover)", value=True)
            run_btn = gr.Button("🚀 शुद्ध आवाज़ बनाएँ", variant="primary")
        
        with gr.Column():
            audio_out = gr.Audio(label="शुद्ध हिंदी आउटपुट")

    run_btn.click(generate_voice, [txt_input, audio_ref, silence_on], audio_out)

demo.launch(share=True)
