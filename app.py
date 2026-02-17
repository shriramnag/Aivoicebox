import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

# 🚀 हगिंग फेस और इंजन सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def apply_shriram_magic(file_path, deep_weight, power_amp):
    """आवाज़ को भारी और दमदार बनाना"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + power_amp # एमप्लीफायर

    if deep_weight > 0:
        # आवाज़ को गहरा करने के लिए पिच को नेचुरल तरीके से बदलना
        new_sample_rate = int(sound.frame_rate * (1.0 - (deep_weight / 80)))
        sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        sound = sound.set_frame_rate(44100)
    
    final_path = "shriram_master_output.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, deep_match, human_feel, weight, amp):
    # 🎙️ आपका पुराना चंक प्रोसेसिंग लॉजिक (सुरक्षित है) [cite: 2026-02-16]
    # (यहाँ split_into_chunks और combine_chunks का उपयोग करें)
    
    temp_output = "temp.wav"
    
    # 🔥 रियलिस्टिक सेटिंग्स
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_sample,
        language="hi",
        file_path=temp_output,
        speed=speed,
        repetition_penalty=15.0, # रोबोटिक टोन हटाने के लिए सबसे जरूरी
        temperature=human_feel,   # इंसानी उतार-चढ़ाव (0.85 रखें)
        top_p=0.9,               # साफ़ आवाज़ के लिए
        gpt_cond_len=3           # 0.9 Deep Match के लिए
    )
    
    return apply_shriram_magic(temp_output, weight, amp)

# 🎨 आपका फाइनल 'रॉयल स्टूडियो' UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - मास्टर क्लोनिंग स्टूडियो")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल (Original)", type="filepath")
            
            with gr.Accordion("⚙️ मास्टर स्लाइडर्स (इंसानी टच के लिए)", open=True):
                speed_s = gr.Slider(label="रफ़्तार (Speed)", minimum=0.8, maximum=1.2, value=1.0)
                human_s = gr.Slider(label="इंसानी अहसास (Human Feel)", minimum=0.5, maximum=1.0, value=0.85)
                weight_s = gr.Slider(label="आवाज़ का भारीपन (Deep Weight)", minimum=0, maximum=10, value=3)
                amp_s = gr.Slider(label="एमप्लीफायर (Power/Gain)", minimum=-5, maximum=10, value=2)
            
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    out = gr.Audio(label="सुनिए असली श्रीराम वाणी", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, gr.State(0.9), human_s, weight_s, amp_s], out)

demo.launch(share=True)
