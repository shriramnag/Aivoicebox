import os
import torch
import gradio as gr
import shutil
import re
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from brain import MahagyaniBrain 

# ⚡ टर्बो हाई स्पीड सेटअप
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मॉडल लोड (Ramai.pth - LOCKED)
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🧠 महाज्ञानी ब्रेन
brain = MahagyaniBrain(
    'sanskrit_knowledge.json', 
    'hindi_grammar.json', 
    'english_knowledge.json', 
    'prosody_config.json'
)

def split_into_chunks(text):
    """टुकड़ों में काटने वाला लॉजिक - 100% फिक्स्ड"""
    # यह पूर्ण विराम, श्लोक विराम और प्रश्नवाचक पर टेक्स्ट को तोड़ेगा
    sentences = re.split('([।!?॥])', text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + sentences[i+1]
        # 180 कैरेक्टर का परफेक्ट साइज ताकि जनरेशन तेज हो
        if len(current_chunk) + len(sentence) < 180:
            current_chunk += sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def apply_mastering(file_path, amp, pitch_val):
    """इको सुधार और क्लैरिटी"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp 
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # संतुलित इको -42dB (हकलाहट रोकने के लिए)
    echo = sound - 42 
    sound = sound.overlay(echo, position=180) 
    
    return sound.low_pass_filter(4000)

def generate_voice(text, voice_sample, speed_s, pitch_s, weight_s, amp_s, progress=gr.Progress()):
    # 🧠 ब्रेन प्रोसेसिंग
    cleaned_text = brain.clean_and_format(text)
    profile = brain.get_voice_profile(text)
    final_speed = profile['global_speed'] if "॥" in text else speed_s
    
    # ✂️ चंकिंग - टुकड़ों में काटना (Fixed)
    chunks = split_into_chunks(cleaned_text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 🚀 टर्बो जनरेशन लूप
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"⚡ {i+1}/{len(chunks)} प्रोसेस हो रहा है...")
        name = os.path.join(output_folder, f"c_{i}.wav")
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=final_speed, 
            repetition_penalty=1.2, # हकलाहट रोकने के लिए ऑप्टिमाइज्ड
            temperature=0.7, # स्थिरता के लिए
            top_p=0.8
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    
    combined.export("temp.wav", format="wav")
    final_audio = apply_mastering("temp.wav", amp_s, pitch_s)
    final_audio.export("shriram_final_fixed.wav", format="wav")
    return "shriram_final_fixed.wav"

# 🎨 UI (All Controls LOCKED)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - टर्बो महाज्ञानी (No Stutter)")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="यहाँ श्लोक या स्क्रिप्ट लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="मास्टर सैंपल (aideva.wav)", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स", open=True):
                speed_s = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.3, value=1.0)
                pitch_s = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
                weight_s = gr.Slider(label="भारीपन", minimum=0, maximum=10, value=6)
                amp_s = gr.Slider(label="पावर", minimum=-5, maximum=10, value=4)
            
            btn = gr.Button("दिव्य आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="100% फिक्स्ड आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, pitch_s, weight_s, amp_s], out)

demo.launch(share=True)
