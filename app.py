import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो हाई स्पीड कॉन्फ़िगरेशन [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True 

# 📥 100% मैच इंजन लोड
print(f"⏳ श्रीराम मास्टर इंजन लोड हो रहा है...")
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, emotion_scale, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 
    
    # 🧹 ऑटो-क्लीनर (0:00 एरर फिक्स)
    output_folder = "outputs"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # ⚡ टर्बो चंकिंग
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 नेचुरल क्लोनिंग: {i+1}/{len(chunks)}") 
        name = os.path.join(output_folder, f"chunk_{i}.wav")
        
        # 🎙️ हकलाहट, चीनी शोर और साँस लेने का फिक्स [cite: 2026-01-06]
        # यहाँ temperature और top_k को "नो-ग्लिच" मोड पर सेट किया गया है
        current_temp = 0.75 + (emotion_scale * 0.05) 
        
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=15.0,   # 👈 हकलाहट रोकने के लिए सबसे सख्त सेटिंग
            temperature=current_temp,  
            top_p=0.85,                # 👈 साँस और नेचुरल फील के लिए
            top_k=25,                  # 👈 चीनी भाषा/अजीब शोर रोकने के लिए लॉक किया गया
            length_penalty=1.0,        
            enable_text_splitting=False 
        )
        chunk_files.append(name)
    
    # ⚡ टर्बो हाई स्पीड मर्जिंग
    final_output = os.path.abspath("shriram_final_pro.wav")
    combine_chunks(chunk_files, output_file=final_output)
    return final_output

# 🎨 श्रीराम वाणी UI (परफेक्ट सेटिंग्स के साथ)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - मास्टर टर्बो v2 (Stable)")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(
                label="स्क्रिप्ट यहाँ लिखें", 
                lines=12, 
                placeholder="साँस और पॉज़ के लिए कोमा (,) या पूर्णविराम (।) का प्रयोग करें और एक स्पेस दें..."
            )
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath", interactive=True)
            
            # 🎚️ टर्बो कंट्रोल्स (लॉक्ड)
            speed_slider = gr.Slider(label="आवाज़ की रफ़्तार (Speed)", minimum=0.5, maximum=1.5, value=1.0, step=0.1)
            pitch_slider = gr.Slider(label="आवाज़ की गहराई (Deep Match)", minimum=0.5, maximum=1.0, value=0.80, step=0.05)
            emotion_slider = gr.Slider(label="इमोशन/साँस की तीव्रता", minimum=0.1, maximum=1.0, value=0.5, step=0.1)
            
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider, emotion_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd(), "/content/"])
