import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो हाई स्पीड कॉन्फ़िगरेशन [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
# GPU (T4) का 100% इस्तेमाल सुनिश्चित करना
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 श्रीराम AI मॉडल लोड
print(f"⏳ टर्बो मोड में मॉडल लोड हो रहा है...")
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 
    
    # टर्बो चंकिंग: बड़े टेक्स्ट को तेज़ी से प्रोसेस करना [cite: 2026-01-06]
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो प्रोसेसिंग: {i+1}/{len(chunks)}") 
        name = os.path.abspath(f"chunk_{i}.wav")
        
        # 🎙️ XTTS टर्बो सेटिंग्स (स्पीड बढ़ाने के लिए) [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=10.0,   # हकलाहट फिक्स
            temperature=pitch,         
            enable_text_splitting=False # मैन्युअल स्प्लिटिंग पहले से है, इसे False रखने से रफ़्तार बढ़ती है
        )
        chunk_files.append(name)
    
    # ⚡ टर्बो हाई स्पीड मर्जिंग
    final_output = os.path.abspath("shriram_final_pro.wav")
    combine_chunks(chunk_files, output_file=final_output)
    
    # प्लेयर फिक्स: पाथ को साफ़ तरीके से वापस भेजना
    return final_output

# 🎨 श्रीराम वाणी UI (आपका ओरिजिनल लुक)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - टर्बो हाई स्पीड v2")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट पेस्ट करें", lines=12, placeholder="यहाँ हिंदी लिखें...")
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath", interactive=True)
            
            # 🎚️ एडवांस कंट्रोल [cite: 2026-01-06]
            speed_slider = gr.Slider(label="आवाज़ की रफ़्तार (Speed)", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
            pitch_slider = gr.Slider(label="आवाज़ की पिच (Pitch)", minimum=0.5, maximum=1.0, value=0.75, step=0.05)
            
            btn = gr.Button("🚀 टर्बो जनरेशन शुरू करें", variant="primary")
            
    with gr.Row():
        # प्लेयर फिक्स: type="filepath" ही रखें ताकि ऑडियो लोड हो सके
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", type="filepath", autoplay=True, interactive=False)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd()])
