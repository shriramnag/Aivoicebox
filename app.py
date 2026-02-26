import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. अल्ट्रा टर्बो सेटअप (LOCKED) [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. हगिंग फेस मॉडल इंटीग्रेशन [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 

print("श्री राम नाग जी, अनलिमिटेड इंजन लोड हो रहा है...")
model_path = hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
hf_hub_download(repo_id=REPO_ID, filename="config.json")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ३. हकलाहट फिक्स टूल [cite: 2026-02-20]
def shiv_super_cleaner(text):
    if not text: return ""
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    text = text.replace('.', ',')
    return text.strip()

# ४. मुख्य इंजन - अनलिमिटेड स्क्रिप्ट कटर (LOCKED) [cite: 2026-01-06]
def generate_shiv_v1_5(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    p_text = shiv_super_cleaner(text)
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ अनलिमिटेड चंकिंग लॉजिक: शब्दों की कोई सीमा नहीं
    all_words = p_text.split()
    chunks = []
    # हर १५० शब्दों पर एक नया हिस्सा बनाना (बिना किसी ४०० की लिमिट के)
    for i in range(0, len(all_words), 150):
        chunks.append(" ".join(all_words[i:i+150]))

    combined = AudioSegment.empty()
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"अनलिमिटेड जनरेशन जारी है... भाग {i+1}")
        
        name = f"turbo_{i}.wav"
        # 🔒 स्टेबिलिटी के लिए सेटिंग [cite: 2026-02-20]
        tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=2.0, temperature=0.5, top_k=50)
        
        seg = AudioSegment.from_wav(name)
        if pitch_s != 1.0:
            new_rate = int(seg.frame_rate * pitch_s)
            seg = seg._spawn(seg.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)

        if use_silence: [cite: 2026-01-06]
            try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=200)
            except: pass
            
        combined += seg
        os.remove(name)
        torch.cuda.empty_cache(); gc.collect()

    if use_clean: [cite: 2026-01-06]
        combined = effects.normalize(combined)
    
    final_output_name = "Shri_Ram_Nag_Output.wav"
    combined.export(final_output_name, format="wav")
    return final_output_name

# ५. UI [cite: 2026-02-20]
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.5 — श्री राम नाग")
    gr.Markdown("### 🔒 अनलिमिटेड स्क्रिप्ट मोड | टर्बो हाई स्पीड [cite: 2026-01-06]")
    
    txt = gr.Textbox(label="लंबी स्क्रिप्ट यहाँ पेस्ट करें (कोई लिमिट नहीं)", lines=15)
    with gr.Row():
        spd = gr.Slider(0.9, 1.4, 1.15, label="रफ़्तार")
        ptch = gr.Slider(0.7, 1.3, 1.0, label="पिच")
    
    btn = gr.Button("🚀 अनलिमिटेड जनरेट और डाउनलोड", variant="primary")
    out = gr.Audio(label="श्री राम नाग आउटपुट", type="filepath")
    
    btn.click(generate_shiv_v1_5, [txt, gr.State(None), gr.State("aideva.wav"), spd, ptch, gr.State(True), gr.State(True)], out)

demo.launch(share=True)
