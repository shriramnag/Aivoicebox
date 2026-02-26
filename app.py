import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप
os.environ["COQUI_TOS_AGREED"] = "1"
torch.backends.cudnn.benchmark = True 
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मॉडल लोड
REPO_ID = "Shriramnag/My-Shriram-Voice" 
print("श्री राम नाग जी, शिव AI का शुद्ध और एरर-फ्री इंजन शुरू हो रहा है...")
try:
    hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
except: pass

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ३. हकलाहट और इंग्लिश फिक्स इंजन
def shiv_super_cleaner(text):
    if not text: return ""
    # इंग्लिश शब्दों को हिंदी उच्चारण में बदलना (हकलाहट रोकने के लिए)
    eng_fix = {
        "Life": "लाइफ", "Dream": "ड्रीम", "Mindset": "माइंडसेट", "Believe": "बिलीव",
        "Success": "सक्सेस", "YouTube": "यूट्यूब", "AI": "ए आई", "Turbo": "टर्बो",
        "Step": "स्टेप", "Fear": "फियर", "Simple": "सिंपल", "Fail": "फेल",
        "Practical": "प्रैक्टिकल", "Strong": "स्ट्रॉन्ग", "Focus": "फोकस"
    }
    for eng, hin in eng_fix.items():
        text = re.sub(rf'\b{eng}\b', hin, text, flags=re.IGNORECASE)

    # नंबरों को शब्दों में बदलना
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    
    # हकलाहट रोकने के लिए ठहराव का इंतजाम
    text = text.replace('.', ', ').replace('।', ', ')
    return text.strip()

# ४. मुख्य इंजन - ४०० टोकन एरर का अंत और अनलिमिटेड लेंथ
def generate_shiv_v1_5(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    p_text = shiv_super_cleaner(text)
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # चंकिंग: ६०-६० शब्दों का सुरक्षित ग्रुप (४०० टोकन एरर फिक्स)
    words = p_text.split()
    chunks = [" ".join(words[i:i+60]) for i in range(0, len(words), 60)]
    combined = AudioSegment.empty()
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"अल्ट्रा-स्मूथ जनरेशन: भाग {i+1}")
        name = f"part_{i}.wav"
        
        # हकलाहट रोकने के लिए ऑप्टिमाइज्ड सेटिंग्स
        tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=2.0, temperature=0.7, top_k=50)
        
        seg = AudioSegment.from_wav(name)
        
        # यहाँ सिंटैक्स एरर को पूरी तरह फिक्स कर दिया गया है
        if use_silence:
            try:
                seg = effects.strip_silence(seg, silence_thresh=-45, padding=300)
            except:
                pass
            
        combined += seg
        os.remove(name)
        torch.cuda.empty_cache(); gc.collect()

    if use_clean:
        combined = combined.set_frame_rate(44100)
        combined = effects.normalize(combined)
    
    final_name = "Shri_Ram_Nag_Output.wav"
    combined.export(final_name, format="wav")
    return final_name

# ५. श्री राम नाग इंटरफेस
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.5 — श्री राम नाग")
    gr.Markdown("### 🔒 १००% एरर फ्री | ३०-४० मिनट मोड | इंग्लिश हकलाहट फिक्स")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="लंबी स्क्रिप्ट यहाँ पेस्ट करें", lines=12)
            word_count = gr.Markdown("शब्द संख्या: **शून्य**")
            txt.change(lambda x: f"शब्द संख्या: **{len(x.split()) if x else 'शून्य'}**", [txt], [word_count])
            
        with gr.Column(scale=1):
            up_v = gr.Audio(label="अपनी आवाज़ अपलोड करें (क्लोनिंग)", type="filepath")
            git_v = gr.Dropdown(choices=["aideva.wav"], label="डिफ़ॉल्ट वॉइस", value="aideva.wav")
            
            with gr.Accordion("⚙️ टर्बो सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.9, 1.4, 1.15, label="रफ़्तार")
                ptch = gr.Slider(0.7, 1.3, 1.0, label="पिच")
                sln = gr.Checkbox(label="Silence Remover", value=True)
                cln = gr.Checkbox(label="Symmetry Clean", value=True)
            
            btn = gr.Button("🚀 १००% शुद्ध जनरेट", variant="primary")
            
    out = gr.Audio(label="श्री राम नाग आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_shiv_v1_5, [txt, up_v, git_v, spd, ptch, sln, cln], out)

demo.launch(share=True)
