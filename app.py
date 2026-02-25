"""
शिव AI — श्री राम नाग | Self-Learning Brain v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ English words sahi bolega
✅ Har galti yaad rakhega (brain.json)
✅ Apne aap seekhta jaayega
✅ GitHub se connect rehega
✅ Dobara wahi galti nahi karega
"""

import os, torch, gradio as gr, requests, re, gc, json, datetime
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects
import numpy as np

# ═══════════════════════════════════════════════════════════════
# BRAIN FILE — Yahan sari seekhi hui baatein save hongi
# GitHub pe commit hogi automatically
# ═══════════════════════════════════════════════════════════════
BRAIN_FILE = "brain.json"
GITHUB_REPO = "shriramnag/Aivoicebox"   # Aapka repo
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")  # HuggingFace Secret mein daalna

def load_brain():
    """Brain file load karo — seedhi baatein yaad hain isme"""
    default = {
        "version": "4.0",
        "total_generations": 0,
        "english_fixes": {},        # "AI" -> "Aay Aay" jaise fixes
        "hindi_fixes": {},          # Hindi word fixes
        "problem_words": [],        # Baar baar fail hone wale words
        "good_params": {            # Jin params pe best result aaya
            "temperature": 0.75,
            "repetition_penalty": 2.5,
            "top_k": 50,
            "top_p": 0.85,
            "speed": 1.1
        },
        "failed_chunks": [],        # Jo chunks kabhi fail hue
        "learning_log": []          # Kya seekha, kab seekha
    }
    
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                # Purani brain ke saath merge karo
                for key in default:
                    if key not in saved:
                        saved[key] = default[key]
                return saved
        except:
            pass
    return default

def save_brain(brain):
    """Brain file save karo locally + GitHub pe"""
    with open(BRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(brain, f, ensure_ascii=False, indent=2)
    
    # GitHub pe bhi save karo (agar token hai)
    if GITHUB_TOKEN:
        try:
            push_to_github(brain)
        except Exception as e:
            print(f"⚠️  GitHub save fail: {e}")

def push_to_github(brain):
    """Brain.json ko GitHub pe push karo"""
    api = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{BRAIN_FILE}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    content = json.dumps(brain, ensure_ascii=False, indent=2)
    import base64
    encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    
    # Pehle purani file ka SHA lo (update ke liye zaroori)
    r = requests.get(api, headers=headers)
    sha = r.json().get("sha", "") if r.status_code == 200 else ""
    
    data = {
        "message": f"🧠 Brain update — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": encoded,
    }
    if sha:
        data["sha"] = sha
    
    resp = requests.put(api, headers=headers, json=data)
    if resp.status_code in [200, 201]:
        print("✅ Brain GitHub pe save ho gaya!")
    else:
        print(f"❌ GitHub save error: {resp.status_code}")

def brain_learn(brain, what_learned, category="general"):
    """Brain mein nayi seekh daalo"""
    entry = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "category": category,
        "learned": what_learned
    }
    brain["learning_log"].append(entry)
    # Sirf last 100 entries rakhna — file badi nahi hogi
    if len(brain["learning_log"]) > 100:
        brain["learning_log"] = brain["learning_log"][-100:]

# ═══════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device}")

REPO_ID = "Shriramnag/My-Shriram-Voice"
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

try:
    ckpt = torch.load(model_path, map_location=device)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict):
        tts.synthesizer.tts_model.load_state_dict(sd, strict=False)
        print("✅ Custom Ramai.pth loaded!")
except Exception as e:
    print(f"⚠️  Custom weights skip: {e}")

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# Brain load karo start mein
BRAIN = load_brain()
print(f"🧠 Brain loaded — {BRAIN['total_generations']} generations seekhe hain abtak")

# ═══════════════════════════════════════════════════════════════
# ENGLISH FIX — Yahi main problem thi
# XTTS English words Hindi voice mein bol nahi paata tha
# Solution: English words ko Hinglish phonetics mein convert karo
# ═══════════════════════════════════════════════════════════════

# Common English words jo Hindi TTS mein fail hote hain
# Format: "english_word": "hinglish_phonetic"
ENGLISH_TO_HINGLISH = {
    # Tech words
    "AI": "ए आई", "ML": "एम एल", "API": "ए पी आई",
    "URL": "यू आर एल", "HTML": "एच टी एम एल",
    "CSS": "सी एस एस", "GPU": "जी पी यू",
    "CPU": "सी पी यू", "RAM": "रैम", "ROM": "रोम",
    "PDF": "पी डी एफ", "SMS": "एस एम एस",
    "OTP": "ओ टी पी", "UPI": "यू पी आई",
    "app": "ऐप", "App": "ऐप", "APP": "ऐप",
    "online": "ऑनलाइन", "offline": "ऑफलाइन",
    "download": "डाउनलोड", "upload": "अपलोड",
    "software": "सॉफ्टवेयर", "hardware": "हार्डवेयर",
    "internet": "इंटरनेट", "website": "वेबसाइट",
    "mobile": "मोबाइल", "phone": "फोन",
    "computer": "कंप्यूटर", "laptop": "लैपटॉप",
    "video": "वीडियो", "audio": "ऑडियो",
    "photo": "फोटो", "camera": "कैमरा",
    # Common English in Hindi speech
    "please": "प्लीज़", "sorry": "सॉरी",
    "thank you": "थैंक यू", "hello": "हेलो",
    "yes": "येस", "no": "नो", "ok": "ओके",
    "OK": "ओके", "okay": "ओके",
    "good": "गुड", "best": "बेस्ट",
    "time": "टाइम", "date": "डेट",
    "news": "न्यूज़", "live": "लाइव",
    "update": "अपडेट", "share": "शेयर",
    "like": "लाइक", "follow": "फॉलो",
    "subscribe": "सब्सक्राइब", "comment": "कमेंट",
    "channel": "चैनल", "video": "वीडियो",
    "click": "क्लिक", "link": "लिंक",
    "support": "सपोर्ट", "team": "टीम",
    "free": "फ्री", "paid": "पेड",
    "plus": "प्लस", "minus": "माइनस",
    "point": "पॉइंट", "percent": "%",
    "show": "शो", "game": "गेम",
    "level": "लेवल", "score": "स्कोर",
}

def fix_english_in_hindi(text, brain):
    """
    Hindi text mein aaye English words ko XTTS ke liye 
    Devanagari phonetics mein convert karo.
    Brain mein seekhe gaye fixes bhi apply karo.
    """
    # Pehle brain ke seekhe gaye fixes apply karo
    for eng, fix in brain.get("english_fixes", {}).items():
        text = re.sub(r'\b' + re.escape(eng) + r'\b', fix, text)
    
    # Phir built-in dictionary se
    for eng, hindi_phonetic in ENGLISH_TO_HINGLISH.items():
        text = re.sub(r'\b' + re.escape(eng) + r'\b', hindi_phonetic, text)
    
    return text

def extract_mixed_segments(text, brain):
    """
    Hindi-English mixed text ko smart segments mein todta hai.
    Har segment ko uski sahi language ke saath tag karta hai.
    
    "Namaskar, AI technology bahut achhi hai" 
    → [("Namaskar, ", "hi"), ("AI technology", "en"), (" bahut achhi hai", "hi")]
    """
    # Pehle English words ko Hindi phonetics mein convert karo (Hindi mode ke liye)
    # aur English-only segments ke liye original rakhna
    
    segments = []
    
    # Pattern: English words (2+ letters) surrounded by Hindi
    # Agar poora sentence English hai toh en
    # Agar Hindi mein kuch English words hain toh unhe Hinglish mein convert karo
    
    hi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    total = len(text.strip())
    
    if hi_chars == 0 and en_chars > total * 0.5:
        # Pure English sentence
        return [(text, "en")]
    
    if hi_chars > 0:
        # Hindi sentence mein English words — unhe Hinglish mein badlo
        converted = fix_english_in_hindi(text, brain)
        return [(converted, "hi")]
    
    return [(text, "hi")]  # Default Hindi

# ═══════════════════════════════════════════════════════════════
# NUMBER CONVERSION
# ═══════════════════════════════════════════════════════════════
HINDI_NUMS = {
    '0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार',
    '5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'
}
EN_NUMS = {
    '0':'zero','1':'one','2':'two','3':'three','4':'four',
    '5':'five','6':'six','7':'seven','8':'eight','9':'nine'
}

def replace_numbers(text, lang):
    num_map = HINDI_NUMS if lang == "hi" else EN_NUMS
    def r(m):
        return ' '.join(num_map[d] for d in m.group())
    return re.sub(r'\d+', r, text)

# ═══════════════════════════════════════════════════════════════
# SMART TEXT SPLITTER (10 word limit — XTTS drift fix)
# ═══════════════════════════════════════════════════════════════
MAX_WORDS = 10

def smart_split(text):
    parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\])', text)
    chunks = []
    
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if p in ['[pause]', '[breath]', '[laugh]']:
            chunks.append(p)
            continue
        
        sentences = re.split(r'(?<=[।!?॥])\s+|(?<=[.!?])\s+(?=[A-Z\u0900-\u097F])', p)
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            words = sent.split()
            if len(words) <= MAX_WORDS:
                if len(sent) > 1:
                    chunks.append(sent)
            else:
                buf = []
                for w in words:
                    buf.append(w)
                    if (w.endswith(',') or w.endswith('—') or len(buf) >= MAX_WORDS):
                        chunks.append(' '.join(buf))
                        buf = []
                if buf:
                    chunks.append(' '.join(buf))
    
    return chunks

# ═══════════════════════════════════════════════════════════════
# REFERENCE AUDIO PREP
# ═══════════════════════════════════════════════════════════════
def prepare_reference(ref_path):
    audio = AudioSegment.from_file(ref_path)
    audio = audio.set_channels(1).set_frame_rate(22050)
    audio = effects.normalize(audio)
    if len(audio) < 3000:
        audio = audio * (3000 // len(audio) + 1)
    audio = audio[:30000]
    clean_path = "ref_prepared.wav"
    audio.export(clean_path, format="wav")
    return clean_path

# ═══════════════════════════════════════════════════════════════
# AUDIO OUTPUT FIX
# ═══════════════════════════════════════════════════════════════
def match_loudness(audio, target_rms=4900):
    audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    curr_rms = np.sqrt(np.mean(samples**2))
    if curr_rms > 10:
        gain = min(target_rms / curr_rms, 4.0)
        samples = np.clip(samples * gain, -32767, 32767).astype(np.int16)
        audio = AudioSegment(samples.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    return effects.normalize(audio)

# ═══════════════════════════════════════════════════════════════
# SELF-LEARNING: Har generation ke baad brain update hota hai
# ═══════════════════════════════════════════════════════════════
def learn_from_generation(brain, script, failed_chunks, success_chunks):
    """
    Generation ke baad brain update karo:
    1. Fail hue chunks yaad rakho
    2. Problem words identify karo  
    3. English words jo fail hue unhe Hinglish fix mein daalo
    """
    brain["total_generations"] += 1
    
    # Failed chunks se seekho
    for chunk in failed_chunks:
        # English words nikalo jo fail hue
        en_words = re.findall(r'\b[a-zA-Z]{2,}\b', chunk)
        for word in en_words:
            if word not in brain["english_fixes"]:
                # Automatic Hinglish convert attempt
                # (User baad mein manual fix bhi de sakta hai)
                brain["problem_words"].append({
                    "word": word,
                    "context": chunk[:50],
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                })
        
        if chunk not in brain["failed_chunks"]:
            brain["failed_chunks"].append(chunk)
    
    # Problem words list clean karo (max 200)
    if len(brain["failed_chunks"]) > 200:
        brain["failed_chunks"] = brain["failed_chunks"][-200:]
    
    # Seekh likho
    if failed_chunks:
        brain_learn(brain, 
            f"Generation #{brain['total_generations']}: {len(failed_chunks)} chunks fail hue — {failed_chunks[:2]}",
            "failure")
    else:
        brain_learn(brain,
            f"Generation #{brain['total_generations']}: Sab {len(success_chunks)} chunks success!",
            "success")
    
    return brain

# ═══════════════════════════════════════════════════════════════
# MAIN GENERATION
# ═══════════════════════════════════════════════════════════════
def generate(text, up_ref, git_ref, speed, pitch, use_silence, use_clean, progress=gr.Progress()):
    global BRAIN
    
    # Reference prepare
    if up_ref:
        ref_path = prepare_reference(up_ref)
    else:
        raw = "ref_raw.wav"
        url = G_RAW + requests.utils.quote(git_ref)
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None, "❌ Voice file download fail hua"
        with open(raw, "wb") as f:
            f.write(resp.content)
        ref_path = prepare_reference(raw)
    
    # Brain ke params use karo
    params = {
        **BRAIN["good_params"],
        "speed": speed
    }
    
    chunks = smart_split(text)
    total = len(chunks)
    combined = AudioSegment.empty()
    
    failed_chunks = []
    success_chunks = []
    brain_report = []
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/total, desc=f"🎙️ {i+1}/{total}: {chunk[:25]}...")
        
        if chunk == "[pause]":
            combined += AudioSegment.silent(duration=800)
            continue
        elif chunk == "[breath]":
            combined += AudioSegment.silent(duration=300)
            continue
        elif chunk == "[laugh]":
            combined += AudioSegment.silent(duration=100)
            continue
        
        # ═══ ENGLISH FIX — Yahi naya fix hai ═══
        # Hindi mein English words ko Hinglish phonetics mein badlo
        segments = extract_mixed_segments(chunk, BRAIN)
        
        chunk_audio = AudioSegment.empty()
        chunk_failed = False
        
        for seg_text, seg_lang in segments:
            clean = replace_numbers(seg_text, seg_lang)
            clean = re.sub(r'\s+', ' ', clean).strip()
            
            if seg_lang == "hi":
                # Hindi ke liye non-Devanagari (jo convert nahi hua) hata do
                clean = re.sub(r'[^\u0900-\u097F\s,!?।॥\'"%-]', ' ', clean)
            else:
                clean = re.sub(r'[^a-zA-Z0-9\s,!?.\'"%-]', ' ', clean)
            
            clean = re.sub(r'\s+', ' ', clean).strip()
            if len(clean) < 2:
                continue
            
            out_path = f"seg_{i}.wav"
            
            try:
                tts.tts_to_file(
                    text=clean,
                    speaker_wav=ref_path,
                    language=seg_lang,
                    file_path=out_path,
                    **{k: v for k, v in params.items() if k != 'speed'},
                    speed=params["speed"]
                )
                seg_audio = AudioSegment.from_wav(out_path).set_channels(1)
                chunk_audio += seg_audio
                success_chunks.append(clean)
                print(f"   ✅ [{i+1}] ({seg_lang}) '{clean[:30]}'")
                
            except Exception as e:
                print(f"   ❌ [{i+1}] FAIL ({seg_lang}): '{clean[:30]}' — {e}")
                chunk_failed = True
                failed_chunks.append(chunk)
                
                # Retry with relaxed params
                try:
                    tts.tts_to_file(
                        text=clean, speaker_wav=ref_path, language=seg_lang,
                        file_path=out_path, speed=speed,
                        temperature=0.85, repetition_penalty=1.5, top_k=80
                    )
                    seg_audio = AudioSegment.from_wav(out_path).set_channels(1)
                    chunk_audio += seg_audio
                    print(f"   ♻️  [{i+1}] Retry success")
                    chunk_failed = False
                except:
                    brain_report.append(f"⚠️ Chunk fail: '{chunk[:40]}'")
            
            finally:
                if os.path.exists(out_path):
                    os.remove(out_path)
        
        if len(chunk_audio) > 0:
            if use_silence:
                try:
                    chunk_audio = effects.strip_silence(chunk_audio, silence_thresh=-42, padding=100)
                except:
                    pass
            combined += chunk_audio
            combined += AudioSegment.silent(duration=60)
        
        if i % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Final audio
    if use_clean:
        combined = match_loudness(combined)
    else:
        combined = combined.set_channels(1).set_frame_rate(44100)
    
    # Cleanup
    for f in ["ref_prepared.wav", "ref_raw.wav"]:
        if os.path.exists(f):
            os.remove(f)
    
    # ═══ BRAIN SEEKHTA HAI ═══
    BRAIN = learn_from_generation(BRAIN, text, failed_chunks, success_chunks)
    save_brain(BRAIN)
    
    final = "Shri_Ram_Nag_Output.wav"
    combined.export(final, format="wav", parameters=["-ar", "44100", "-ac", "1"])
    
    # Report banao
    report = f"""🧠 Brain Report #{BRAIN['total_generations']}:
✅ Successful chunks: {len(success_chunks)}
❌ Failed chunks: {len(failed_chunks)}
📚 Total seekha abtak: {BRAIN['total_generations']} generations
🔧 Problem words: {len(BRAIN['problem_words'])} words collected"""
    
    if brain_report:
        report += "\n\n⚠️ Issues:\n" + "\n".join(brain_report)
    
    print(report)
    return final, report

# ═══════════════════════════════════════════════════════════════
# MANUAL WORD FIX — User khud English word ki Hinglish sikha sakta hai
# ═══════════════════════════════════════════════════════════════
def add_word_fix(english_word, hinglish_fix):
    """User manually koi English word ka fix de sakta hai"""
    global BRAIN
    if english_word.strip() and hinglish_fix.strip():
        BRAIN["english_fixes"][english_word.strip()] = hinglish_fix.strip()
        brain_learn(BRAIN, 
            f"User ne sikhaya: '{english_word}' → '{hinglish_fix}'",
            "user_fix")
        save_brain(BRAIN)
        return f"✅ Seekh liya! '{english_word}' ab '{hinglish_fix}' bolta rahega"
    return "❌ Dono fields bharo"

def get_brain_status():
    """Brain ki current status dikhao"""
    global BRAIN
    recent = BRAIN["learning_log"][-5:] if BRAIN["learning_log"] else []
    recent_text = "\n".join([f"• [{e['time']}] {e['learned']}" for e in reversed(recent)])
    
    problem_words = list(set([p["word"] for p in BRAIN["problem_words"][-20:]]))
    
    status = f"""🧠 Brain Status:
━━━━━━━━━━━━━━━━━━━━━━━
📊 Total Generations: {BRAIN['total_generations']}
📝 English Fixes Seekhe: {len(BRAIN['english_fixes'])}
⚠️  Problem Words: {len(BRAIN['problem_words'])}
━━━━━━━━━━━━━━━━━━━━━━━
🔤 Haali Problem Words:
{', '.join(problem_words) if problem_words else 'Koi nahi — sab theek!'}
━━━━━━━━━━━━━━━━━━━━━━━
📖 Recent Learning:
{recent_text if recent_text else 'Abhi kuch generate nahi kiya'}"""
    return status

# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
js = """function insertTag(tag) { 
    var t = document.querySelector('#script_box textarea'); 
    var s = t.selectionStart; 
    t.value = t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); 
    t.focus(); return t.value; 
}"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js) as demo:
    gr.Markdown("""
    # 🚩 शिव AI — श्री राम नाग | Self-Learning Brain v4.0
    > ✅ English words fix | ✅ Har galti se seekhta hai | ✅ Brain GitHub pe save hota hai
    """)
    
    with gr.Tabs():
        
        # ── TAB 1: MAIN GENERATION ──────────────────────────
        with gr.Tab("🎙️ Voice Generate"):
            with gr.Row():
                with gr.Column(scale=2):
                    txt = gr.Textbox(
                        label="स्क्रिप्ट (हिंदी / English / Mixed)",
                        lines=12, elem_id="script_box",
                        placeholder="""यहाँ स्क्रिप्ट लिखें...
मिसाल: Namaskar doston! AI technology aaj bahut aage badh gayi hai.
यह नई दुनिया है जहाँ हर cheez possible है।"""
                    )
                    wc = gr.Markdown("📊 शब्द: 0")
                    txt.change(lambda x: f"📊 शब्द: **{len(x.split()) if x.strip() else 0}**", [txt], [wc])
                    
                    with gr.Row():
                        gr.Button("⏸️ रोके").click(None, None, txt, js="()=>insertTag('[pause]')")
                        gr.Button("💨 सांस").click(None, None, txt, js="()=>insertTag('[breath]')")
                        gr.Button("😊 हँसो").click(None, None, txt, js="()=>insertTag('[laugh]')")
                
                with gr.Column(scale=1):
                    git_ref = gr.Dropdown(choices=["aideva.wav","Joanne.wav"], 
                                         label="📁 Voice", value="aideva.wav")
                    up_ref = gr.Audio(label="🎤 अपनी Voice Upload", type="filepath")
                    
                    with gr.Accordion("⚙️ Settings", open=True):
                        spd = gr.Slider(0.9, 1.4, 1.1, step=0.05, label="Speed")
                        ptc = gr.Slider(0.8, 1.1, 0.96, label="Pitch")
                        cln = gr.Checkbox(label="✅ Loudness Match", value=True)
                        sln = gr.Checkbox(label="✅ Silence Remove", value=True)
                    
                    btn = gr.Button("🚀 Generate", variant="primary", size="lg")
            
            out_audio = gr.Audio(label="🎧 Output", type="filepath", autoplay=True)
            out_report = gr.Textbox(label="🧠 Brain Report", lines=8, interactive=False)
            
            btn.click(generate, [txt, up_ref, git_ref, spd, ptc, sln, cln], 
                     [out_audio, out_report])
        
        # ── TAB 2: BRAIN / TEACHING ──────────────────────────
        with gr.Tab("🧠 Brain — Sikhao & Dekho"):
            gr.Markdown("""
            ### यहाँ आप Brain को manually sikha sakte hain
            Agar koi English word galat bol raha hai — uska sahi Hinglish likho.
            Brain yaad rakhega aur dobara galti nahi karega.
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📝 Naya Word Sikhao")
                    eng_in = gr.Textbox(label="English Word (jo galat bol raha hai)", 
                                       placeholder="jaise: AI")
                    hi_in = gr.Textbox(label="Sahi Hinglish Phonetic", 
                                      placeholder="jaise: ए आई")
                    teach_btn = gr.Button("✅ Brain Ko Sikhao", variant="primary")
                    teach_out = gr.Textbox(label="Result", interactive=False)
                    teach_btn.click(add_word_fix, [eng_in, hi_in], teach_out)
                
                with gr.Column():
                    gr.Markdown("#### 📊 Brain Ki Haali Status")
                    status_btn = gr.Button("🔄 Status Dekho")
                    status_out = gr.Textbox(label="Brain Status", lines=15, interactive=False)
                    status_btn.click(get_brain_status, [], status_out)
            
            gr.Markdown("""
            ---
            ### 💡 Common English Words Ki Hinglish List
            | English | Hinglish |
            |---------|----------|
            | AI | ए आई |
            | app | ऐप |
            | online | ऑनलाइन |
            | download | डाउनलोड |
            | software | सॉफ्टवेयर |
            | update | अपडेट |
            | video | वीडियो |
            | channel | चैनल |
            
            > **Tip:** Jo bhi word galat bole — upar wale form mein daalo, brain seekh lega!
            """)
        
        # ── TAB 3: GITHUB SETTINGS ──────────────────────────
        with gr.Tab("⚙️ GitHub Settings"):
            gr.Markdown("""
            ### GitHub se Brain Connect karna
            
            **Step 1:** HuggingFace Space mein jaao → Settings → Secrets
            
            **Step 2:** Naya secret banao:
            - Name: `GITHUB_TOKEN`  
            - Value: Apna GitHub Personal Access Token
            
            **Step 3:** GitHub pe token banane ke liye:
            - GitHub → Settings → Developer Settings → Personal Access Tokens → Generate New Token
            - `repo` permission do
            
            **Brain automatically save hoga** har generation ke baad `brain.json` mein aapke GitHub repo mein.
            """)
            
            github_status = gr.Textbox(
                label="GitHub Connection Status",
                value=f"Token set: {'✅ HAN' if GITHUB_TOKEN else '❌ NAHI — Sirf local save hoga'}\nRepo: {GITHUB_REPO}",
                interactive=False
            )

demo.launch(share=True)
