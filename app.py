import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मॉडल लोड
print(f"⏳ इंजन गरम हो रहा है...")
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों दें।") 
    
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress(i/len(chunks), desc=f"वाक्य {i+1} प्रोसेस हो रहा है...") 
        name = os.path.abspath(f"chunk_{i}.wav")
        
        # 🎙️ हकलाहट रोकने और शुद्ध क्लोनिंग के लिए बैकएंड फिक्स [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            repetition_penalty=5.0 # हकलाना बंद करने के लिए
        )
        chunk_files.append(name)
    
    # टुकड़ों को जोड़ना
    final_output = combine_chunks(chunk_files)
    return os.path.abspath(final_output)

# 🎨 श्रीराम वाणी - ओरिजिनल लुक
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी - टर्बो v2") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - टर्बो v2")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="10,000 पोतीस तक की कहानी", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस विवरण (wav)", type="filepath", interactive=True)
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="अंतिम आवाज़", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd()])
