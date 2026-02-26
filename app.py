import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप (LOCKED) [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
torch.backends.cudnn.benchmark = True 
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मॉडल लोड [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
try:
    hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
    hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")
except: pass

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ३. हकलाहट और इंग्लिश शब्दों का पक्का सुधार [cite: 2026-02-20]
def shiv_super_cleaner(text):
    if not text: return ""
    eng_fix = {
        "Life": "लाइफ", "Dream": "ड्रीम", "Mindset": "माइंडसेट", "Believe": "बिलीव",
        "Success": "सक्सेस", "YouTube": "यूट्यूब", "AI": "ए आई", "Turbo": "टर्बो",
        "Strong": "स्ट्रॉन्ग", "Step": "स्टेप", "Fear": "फियर", "Simple": "सिंपल",
        "Practical": "प्रैक्टिकल", "Practice": "प्रैक्टिस", "Focus": "फोकस", "Improvement": "इंप्रूवमेंट"
    }
    for eng, hin in eng_fix.items():
        text = re.sub(rf'\b{eng}\b', hin, text, flags=re.IGNORECASE)

    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    
    text = text.replace('.', ',')
    return text.strip()

# ४. मुख्य इंजन - क्लोनिंग + अनलिमिटेड लेंथ (LOCKED) [cite: 2026-02-26]
def generate_shiv_v1_5(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    p_text = shiv_super_cleaner(text)
    
    # वॉइस क्लोनिंग लॉजिक: अगर फाइल अपलोड की है तो वही इस्तेमाल होगी [cite: 2026-02-22]
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ४०० टोकन एरर फिक्स के लिए ७० शब्दों का सुरक्षित चंक [cite: 2026-02-26]
    words = p_text.split()
    chunks = [" ".join(words[i:i+70]) for i in range(0, len(words), 70)]
    combined = AudioSegment.empty()
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"जनरेशन जारी है... भाग {i+1}")
        name = f"part_{i}.wav"
        
        # हकलाहट रोकने के लिए सेटिंग्स [cite: 2026-02-20]
        tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=2.0, temperature=0.6, top_k=50)
        
        seg = AudioSegment.from_wav(name)
        if use_silence:
            try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=200)
            except: pass
            
        combined += seg
        os.remove(name)
        torch.cuda.empty_cache(); gc.collect()

    if use_clean:
        combined = effects.normalize(combined)
    
    final_name = "Shri_Ram_Nag_Output.wav"
    combined.export(final_name, format="wav")
    return final_name

# ५. श्री राम नाग स्पेशल यूआई (LOCKED)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.5 — श्री राम नाग")
    gr.Markdown("### 🔒 क्लोनिंग + अनलिमिटेड मोड | ४०० टोकन फिक्स | हकलाहट फिक्स")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="लंबी स्क्रिप्ट यहाँ पेस्ट करें", lines=12)
            word_count = gr.Markdown("शब्द संख्या: **शून्य**")
            txt.change(lambda x: f"शब्द संख्या: **{len(x.split()) if x else 'शून्य'}**", [txt], [word_count])
            
        with gr.Column(scale=1):
            # वॉइस क्लोनिंग अपलोड क्षेत्र वापस जोड़ दिया गया है [cite: 2026-02-22]
            up_v = gr.Audio(label="अपनी आवाज़ अपलोड करें (क्लोनिंग के लिए)", type="filepath")
            git_v = gr.Dropdown(choices=["aideva.wav"], label="या डिफ़ॉल्ट वॉइस चुनें", value="aideva.wav")
            
            with gr.Accordion("⚙️ टर्बो सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.9, 1.4, 1.15, label="रफ़्तार")
                ptch = gr.Slider(0.7, 1.3, 1.0, label="पिच")
                sln = gr.Checkbox(label="Silence Remover", value=True)
                cln = gr.Checkbox(label="Symmetry Clean", value=True)
            
            btn = gr.Button("🚀 टर्बो जनरेट", variant="primary")
            
    out = gr.Audio(label="श्री राम नाग आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_shiv_v1_5, [txt, up_v, git_v, spd, ptch, sln, cln], out)

demo.launch(share=True)
