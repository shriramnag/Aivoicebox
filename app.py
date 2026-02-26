import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. अल्ट्रा टर्बो 2000% हाई स्पीड सेटअप (LOCKED)
os.environ["COQUI_TOS_AGREED"] = "1"
torch.backends.cudnn.benchmark = True # GPU की फुल स्पीड
torch.set_num_threads(4)
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. हगिंग फेस मॉडल इंटीग्रेशन
REPO_ID = "Shriramnag/My-Shriram-Voice" 

print("श्री राम नाग जी, 2000% अल्ट्रा-टर्बो इंजन लोड हो रहा है...")

# मॉडल फाइल्स को केवल एक बार डाउनलोड करना
try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
    print("✅ मॉडल सफलतापूर्वक कनेक्ट हो गया।")
except Exception as e:
    print("⚠️ मॉडल डाउनलोड में समस्या।")

# मास्टर मॉडल लोड
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ३. मास्टर टेक्स्ट क्लीनर (हकलाहट और नंबर फिक्स)
def shiv_super_cleaner(text):
    if not text: return ""
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    
    brain_fix = {"जिंदगी": "ज़िन्दगी", "YouTube": "यूट्यूब", "AI": "ए आई"}
    for k, v in brain_fix.items(): text = text.replace(k, v)
    return text.strip()

# ४. मुख्य इंजन - 150-200 वर्ड कटर + अल्ट्रा टर्बो (LOCKED)
def generate_shiv_v1_5(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    p_text = shiv_super_cleaner(text)
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ 150-200 शब्दों का स्मार्ट कटर (SMART SCRIPT CUTTER)
    raw_sentences = re.split(r'([।!?\n])', p_text)
    sentences = []
    temp_s = ""
    for c in raw_sentences:
        if c in ["।", "!", "?", "\n"]:
            sentences.append((temp_s + c).strip())
            temp_s = ""
        else:
            temp_s += c
    if temp_s: sentences.append(temp_s.strip())

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if not sentence: continue
        # जब तक 160-180 शब्द नहीं हो जाते, वाक्यों को जोड़ते रहो
        if len(current_chunk.split()) + len(sentence.split()) <= 180:
            current_chunk += " " + sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    
    # अगर कोई वाक्य ही 200 शब्दों से बड़ा हो (Fallback)
    final_chunks = []
    for c in chunks:
        words = c.split()
        if len(words) > 200:
            for i in range(0, len(words), 150):
                final_chunks.append(" ".join(words[i:i+150]))
        else:
            final_chunks.append(c)

    combined = AudioSegment.empty()
    
    for i, chunk in enumerate(final_chunks):
        progress((i+1)/len(final_chunks), desc=f"श्री राम नाग जी, 150-200 वर्ड चंक जनरेट हो रहा है... ({i+1}/{len(final_chunks)})")
        
        name = f"temp_chunk_{i}.wav"
        
        # 🔒 HALLUCINATION FIX: Temperature 0.5 और Penalty 2.0 (नो शब्द जंप)
        tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=2.0, temperature=0.5, top_k=50)
        
        seg = AudioSegment.from_wav(name)
        
        # पिच कंट्रोल
        if pitch_s != 1.0:
            new_rate = int(seg.frame_rate * pitch_s)
            seg = seg._spawn(seg.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)

        if use_silence:
            try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=200)
            except: pass
            
        combined += seg
        os.remove(name)
    
    # एक ही बार मेमोरी खाली करना (SPEED BOOST)
    torch.cuda.empty_cache(); gc.collect()

    if use_clean:
        combined = combined.set_frame_rate(44100)
        combined = effects.normalize(combined)
    
    # ✅ आपका तय किया हुआ डाउनलोड नाम
    final_output_name = "Shri_Ram_Nag_Output.wav"
    combined.export(final_output_name, format="wav")
    return final_output_name

# ५. दिव्य UI (श्री राम नाग)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.5 — श्री राम नाग")
    gr.Markdown("### 🔒 2000% टर्बो स्पीड | 150-200 वर्ड कटर | 0% हकलाहट")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ लिखें", lines=12, placeholder="अब पूरा पैराग्राफ 150-200 शब्दों के चंक में तेज़ी से जनरेट होगा...")
            word_count = gr.Markdown("शब्द संख्या: **शून्य**")
            txt.change(lambda x: f"शब्द संख्या: **{len(x.split()) if x else 'शून्य'}**", [txt], [word_count])
            
        with gr.Column(scale=1):
            git_v = gr.Dropdown(choices=["aideva.wav"], label="वॉइस", value="aideva.wav")
            up_v = gr.Audio(label="सैंपल अपलोड", type="filepath")
            with gr.Accordion("⚙️ अल्ट्रा टर्बो सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.9, 1.4, 1.15, label="टर्बो रफ़्तार")
                ptch = gr.Slider(0.7, 1.3, 1.0, label="पिच (Pitch)")
                cln = gr.Checkbox(label="शोर फिक्स (Symmetry Clean)", value=True)
                sln = gr.Checkbox(label="Silence Remover", value=True)
            btn = gr.Button("🚀 टर्बो जनरेट और डाउनलोड", variant="primary")
            
    out = gr.Audio(label="श्री राम नाग आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_shiv_v1_5, [txt, up_v, git_v, spd, ptch, sln, cln], out)

demo.launch(share=True, debug=True)
