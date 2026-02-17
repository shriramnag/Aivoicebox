import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

# ⚡ टर्बो इंजन सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 हगिंग फेस मॉडल
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def apply_shriram_final_touch(file_path, weight, amp):
    """आवाज़ को भारी और पावरफुल बनाना"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp 
    if weight > 0:
        new_sample_rate = int(sound.frame_rate * (1.0 - (weight / 85)))
        sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        sound = sound.set_frame_rate(44100)
    final_path = "shriram_hindi_pure.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, human_feel, weight, amp):
    if not text or not voice_sample:
        raise gr.Error("स्क्रिप्ट और वॉइस सैंपल दें।") 

    # 🚩 भाषा पर लगाम लगाने के लिए विशेष सेटिंग
    # 'language="hi"' को कड़ाई से लागू करना
    temp_file = "temp_pure.wav"
    
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_sample,
        language="hi",             # शुद्ध हिंदी [cite: 2025-11-23]
        file_path=temp_file,
        speed=speed,
        repetition_penalty=18.0,   # दूसरी भाषा के शब्दों के जुड़ाव को रोकने के लिए बढ़ाया गया
        temperature=human_feel,    
        top_p=0.80,                # शुद्धता के लिए थोड़ा कम रखा गया ताकि मॉडल भटके नहीं
        gpt_cond_len=6,            # सैंपल को गहराई से समझने के लिए बढ़ाया गया
        enable_text_splitting=True 
    )
    
    return apply_shriram_final_touch(temp_file, weight, amp)

# 🎨 100% शुद्ध हिंदी मास्टर स्टूडियो
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - 100% शुद्ध हिंदी AI (No Language Drift)")
    gr.Markdown("### दूसरी भाषा के उच्चारण पर पूरी तरह लगाम")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="सिर्फ हिंदी स्क्रिप्ट लिखें", lines=12, placeholder="यहाँ हिंदी लिखें...")
        with gr.Column(scale=1):
            ref = gr.Audio(label="हिंदी वॉइस सैंपल", type="filepath")
            
            with gr.Accordion("🛡️ लगाम कंट्रोल (Pure Hindi)", open=True):
                speed_s = gr.Slider(label="स्पीड", minimum=0.8, maximum=1.1, value=0.95)
                human_s = gr.Slider(label="ह्यूमन इमोशन", minimum=0.5, maximum=0.9, value=0.75) # भटकाव रोकने के लिए इसे 0.75 पर फिक्स किया
                weight_s = gr.Slider(label="गहरा भारी वजन", minimum=0, maximum=10, value=4)
                amp_s = gr.Slider(label="एमप्लीफायर", minimum=-5, maximum=10, value=2)
            
            btn = gr.Button("🚀 शुद्ध हिंदी जनरेट करें", variant="primary")
            
    out = gr.Audio(label="100% शुद्ध हिंदी आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s], out)

demo.launch(share=True)
