import os
import torch
import gradio as gr
import shutil
import re
import gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from brain import MahagyaniBrain 

# ⚡ टर्बो हाई स्पीड & GPU लॉक [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🧠 महाज्ञानी ब्रेन
brain = MahagyaniBrain(
    'sanskrit_knowledge.json', 'hindi_grammar.json', 
    'english_knowledge.json', 'prosody_config.json'
)

def permanent_number_fix(text):
    """NotImplementedError को हमेशा के लिए खत्म करने के लिए"""
    # नंबरों को शब्दों में बदलने का सुरक्षित तरीका
    num_map = {
        '0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच',
        '6':'छह','7':'सात','8':'आठ','9':'नौ'
    }
    for num, word in num_map.items():
        text = text.replace(num, word)
    return text

def count_words(text):
    """वर्ड काउंटर लॉजिक [cite: 2026-02-18]"""
    if not text: return "शब्द: 0"
    words = len(text.split())
    return f"शब्द: {words}"

def split_into_chunks(text):
    """चंकिंग लॉजिक - LOCKED [cite: 2026-02-18]"""
    sentences = re.split('([।!?॥\n])', text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + sentences[i+1]
        if len(current_chunk) + len(sentence) < 150:
            current_chunk += sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c) > 2]

def generate_voice(text, voice_sample, speed_s, pitch_s, weight_s, amp_s, progress=gr.Progress()):
    # 1. परमानेंट एरर फिक्स और ब्रेन प्रोसेसिंग
    text = permanent_number_fix(text) 
    cleaned_text = brain.clean_and_format(text)
    profile = brain.get_voice_profile(text)
    final_speed = profile['global_speed'] if "॥" in text else speed_s
    
    # 2. चंकिंग और प्रोग्रेस गिनती
    chunks = split_into_chunks(cleaned_text)
    total = len(chunks)
    chunk_files = []
    output_folder = "turbo_cache"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    combined = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        progress((i+1)/total, desc=f"🚀 टर्बो जनरेशन: भाग {i+1} / {total}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=final_speed, temperature=0.75, repetition_penalty=5.0
        )
        combined += AudioSegment.from_wav(name)
        if i % 5 == 0: torch.cuda.empty_cache(); gc.collect()

    final_path = "shriram_fixed_final.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 UI डिज़ाइन (Word Counter के साथ)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - महाज्ञानी टर्बो (LOCKED)")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ पेस्ट करें", lines=12)
            word_count_display = gr.Label(value="शब्द: 0", label="काउंटर")
            # टेक्स्ट बदलते ही शब्दों को गिनना
            txt.change(count_words, inputs=[txt], outputs=[word_count_display])
            
        with gr.Column(scale=1):
            ref = gr.Audio(label="मास्टर सैंपल", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                speed_s = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.4, value=1.0)
                pitch_s = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
                weight_s = gr.Slider(label="भारीपन", minimum=0, maximum=10, value=6)
                amp_s = gr.Slider(label="शक्ति", minimum=-5, maximum=10, value=4)
            btn = gr.Button("दिव्य टर्बो जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="100% शुद्ध आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, pitch_s, weight_s, amp_s], out)

demo.launch(share=True)
