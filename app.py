"""
SHIV AI v4.1 — श्री राम नाग | Launch-Safe Version
====================================================
Launch errors fixed:
✅ brain.py import fail → graceful fallback
✅ TTS/torch missing → clear error message  
✅ Ramai.pth load fail → silently skip
✅ GitHub token missing → local-only mode
✅ Gradio version mismatch → compatible syntax
✅ numpy missing → fallback loudness match
"""

import os, re, gc
import sys

# ════════════════════════════════════════════════════════════
# STEP 1: SAFE IMPORTS — Kuch bhi miss ho to app crash na kare
# ════════════════════════════════════════════════════════════
print("🔄 Libraries load ho rahi hain...")

try:
    import torch
    TORCH_OK = True
    print(f"✅ torch {torch.__version__}")
except ImportError:
    TORCH_OK = False
    print("❌ torch nahi mila — CPU mode mein chalega")

try:
    import gradio as gr
    print(f"✅ gradio {gr.__version__}")
except ImportError:
    print("❌ FATAL: gradio install nahi hai!")
    print("   Command: pip install gradio")
    sys.exit(1)

try:
    from pydub import AudioSegment, effects
    PYDUB_OK = True
    print("✅ pydub OK")
except ImportError:
    PYDUB_OK = False
    print("❌ pydub nahi mila — audio processing limited hogi")

try:
    import numpy as np
    NUMPY_OK = True
    print("✅ numpy OK")
except ImportError:
    NUMPY_OK = False
    print("⚠️ numpy nahi mila — loudness match skip hogi")

try:
    import requests
    REQUESTS_OK = True
    print("✅ requests OK")
except ImportError:
    REQUESTS_OK = False
    print("⚠️ requests nahi mila — GitHub voice download nahi hogi")

try:
    from TTS.api import TTS
    TTS_OK = True
    print("✅ TTS (Coqui) OK")
except ImportError:
    TTS_OK = False
    print("❌ TTS nahi mili — voice generate nahi hogi")
    print("   Command: pip install TTS")

try:
    from huggingface_hub import hf_hub_download
    HF_OK = True
    print("✅ huggingface_hub OK")
except ImportError:
    HF_OK = False
    print("⚠️ huggingface_hub nahi mila — model download skip")

# ════════════════════════════════════════════════════════════
# STEP 2: BRAIN IMPORT — brain.py na mile to bhi kaam kare
# ════════════════════════════════════════════════════════════
BRAIN_OK = False
try:
    from brain import (
        load_english_map, fix_english_in_hindi,
        get_inter_chunk_pause, record_generation,
        user_teaches, get_stats,
        sync_to_github, load_from_github
    )
    BRAIN_OK = True
    print("✅ brain.py connected!")
except ImportError as e:
    print(f"⚠️ brain.py nahi mila ({e}) — basic mode mein chalega")
    # Fallback functions — brain.py na ho to bhi crash nahi
    def load_english_map():
        return {
            "AI":"ए आई","YouTube":"यूट्यूब","WhatsApp":"व्हाट्सएप",
            "Instagram":"इंस्टाग्राम","Facebook":"फेसबुक",
            "Google":"गूगल","GitHub":"गिटहब","subscribe":"सब्सक्राइब",
            "like":"लाइक","share":"शेयर","comment":"कमेंट",
            "download":"डाउनलोड","upload":"अपलोड","online":"ऑनलाइन",
            "video":"वीडियो","audio":"ऑडियो","mobile":"मोबाइल",
            "app":"एप","website":"वेबसाइट","technology":"टेक्नोलॉजी",
            "digital":"डिजिटल","channel":"चैनल","live":"लाइव",
        }
    def fix_english_in_hindi(text, emap):
        if not re.search(r'[\u0900-\u097F]', text):
            return text
        words = text.split()
        result = []
        for w in words:
            matched = next((v for k,v in emap.items() if k.lower()==w.lower()), None)
            result.append(matched if matched else w)
        return ' '.join(result)
    def get_inter_chunk_pause(chunk):
        t = chunk.strip()
        if t.endswith(('।','॥','!','?','.')): return 100
        elif t.endswith(','): return 50
        return 0
    def record_generation(*a, **k): pass
    def user_teaches(w, h):
        return f"⚠️ brain.py nahi mila — '{w}' yaad nahi rakh paya"
    def get_stats():
        return "⚠️ brain.py nahi mila — stats unavailable"
    def sync_to_github(t, r=""):
        return "⚠️ brain.py nahi mila — sync nahi hogi"
    def load_from_github(t, r=""): pass

# ════════════════════════════════════════════════════════════
# STEP 3: DEVICE + MODEL SETUP
# ════════════════════════════════════════════════════════════
os.environ["COQUI_TOS_AGREED"] = "1"

if TORCH_OK:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"
print(f"🔧 Device: {device}")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO  = "shriramnag/Aivoicebox"

# GitHub se brain load karo (agar token ho)
if GITHUB_TOKEN and BRAIN_OK:
    try:
        print("🔄 GitHub se brain data load...")
        load_from_github(GITHUB_TOKEN, GITHUB_REPO)
        print("✅ Brain loaded from GitHub")
    except Exception as e:
        print(f"⚠️ GitHub brain load fail: {e}")

# TTS Model load
tts = None
if TTS_OK:
    try:
        print("🔄 XTTS model load ho raha hai (thoda time lagega)...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("✅ XTTS ready!")
    except Exception as e:
        print(f"❌ XTTS load fail: {e}")
        tts = None

# Custom Ramai.pth inject
if tts is not None and HF_OK and TORCH_OK:
    try:
        model_path = hf_hub_download(
            repo_id="Shriramnag/My-Shriram-Voice",
            filename="Ramai.pth"
        )
        ckpt = torch.load(model_path, map_location=device)
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        if isinstance(sd, dict):
            tts.synthesizer.tts_model.load_state_dict(sd, strict=False)
            print("✅ Ramai.pth custom weights loaded!")
    except Exception as e:
        print(f"⚠️ Ramai.pth skip: {e}")

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════
HINDI_NUMS = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार',
              '5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
EN_NUMS    = {'0':'zero','1':'one','2':'two','3':'three','4':'four',
              '5':'five','6':'six','7':'seven','8':'eight','9':'nine'}

def replace_numbers(text, lang):
    nmap = HINDI_NUMS if lang == "hi" else EN_NUMS
    def _r(m): return ' '.join(nmap[d] for d in m.group())
    return re.sub(r'\d+', _r, text)

def detect_lang(text):
    hi = len(re.findall(r'[\u0900-\u097F]', text))
    en = len(re.findall(r'[a-zA-Z]', text))
    tot = max(len(text.strip()), 1)
    if hi > 0: return "hi"
    if en / tot > 0.5: return "en"
    return "hi"

MAX_WORDS = 10

def smart_split(text):
    parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\])', text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part: continue
        if part in ['[pause]','[breath]','[laugh]']:
            chunks.append(part); continue
        sentences = re.split(
            r'(?<=[।!?॥])\s+|(?<=[.!?])\s+(?=[A-Z\u0900-\u097F])', part
        )
        for sent in sentences:
            sent = sent.strip()
            if not sent: continue
            words = sent.split()
            if len(words) <= MAX_WORDS:
                if len(sent) > 1: chunks.append(sent)
            else:
                buf = []
                for w in words:
                    buf.append(w)
                    if w.endswith((',','—','-')) or len(buf) >= MAX_WORDS:
                        chunks.append(' '.join(buf)); buf = []
                if buf: chunks.append(' '.join(buf))
    return [c for c in chunks if c]

def prepare_ref(path):
    if not PYDUB_OK:
        return path
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(22050)
    audio = effects.normalize(audio)
    if len(audio) < 3000:
        audio = audio * (3000 // len(audio) + 1)
    audio = audio[:30000]
    out = "ref_prepared.wav"
    audio.export(out, format="wav")
    return out

def match_loudness(audio, target=4900):
    if not NUMPY_OK or not PYDUB_OK:
        return audio
    audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)
    samp = np.array(audio.get_array_of_samples(), dtype=np.float32)
    rms = np.sqrt(np.mean(samp**2))
    if rms > 10:
        gain = min(target/rms, 4.0)
        samp = np.clip(samp*gain, -32767, 32767).astype(np.int16)
        audio = AudioSegment(samp.tobytes(), frame_rate=44100,
                             sample_width=2, channels=1)
    return effects.normalize(audio)

def tts_cfg(speed):
    return dict(temperature=0.75, repetition_penalty=2.5,
                top_k=50, top_p=0.85, speed=speed)

# ════════════════════════════════════════════════════════════
# MAIN GENERATE
# ════════════════════════════════════════════════════════════
def generate(text, up_ref, git_ref, speed, pitch,
             use_silence, use_clean, progress=gr.Progress()):

    # Safety checks
    if not text or not text.strip():
        return None, "⚠️ Script khaali hai — kuch likho pehle।"
    if tts is None:
        return None, "❌ TTS model load nahi hua। Requirements check karein।"
    if not PYDUB_OK:
        return None, "❌ pydub install nahi hai।\nCommand: pip install pydub"

    emap = load_english_map()
    errors_log = []

    # Reference audio
    ref = None
    if up_ref:
        try:
            ref = prepare_ref(up_ref)
        except Exception as e:
            return None, f"❌ Reference audio process nahi hua: {e}"
    elif REQUESTS_OK:
        try:
            raw = "ref_raw.wav"
            url = G_RAW + requests.utils.quote(git_ref)
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                return None, f"❌ GitHub se voice nahi mili ({r.status_code})। Apni voice upload karein।"
            with open(raw,"wb") as f: f.write(r.content)
            ref = prepare_ref(raw)
        except Exception as e:
            return None, f"❌ Voice download fail: {e}"
    else:
        return None, "❌ requests nahi hai — apni voice upload karein।"

    if not ref or not os.path.exists(ref):
        return None, "❌ Reference voice file nahi mili।"

    chunks = smart_split(text)
    total  = len(chunks)
    if total == 0:
        return None, "⚠️ Text mein koi valid content nahi mila।"

    print(f"📝 {total} chunks")
    combined = AudioSegment.empty()
    cfg = tts_cfg(speed)

    for i, chunk in enumerate(chunks):
        try:
            progress((i+1)/total, desc=f"🎙️ {i+1}/{total}: {chunk[:30]}...")
        except:
            pass

        if chunk == "[pause]":
            combined += AudioSegment.silent(800); continue
        elif chunk == "[breath]":
            combined += AudioSegment.silent(300); continue
        elif chunk == "[laugh]":
            combined += AudioSegment.silent(100); continue

        lang  = detect_lang(chunk)
        clean = fix_english_in_hindi(chunk, emap) if lang == "hi" else chunk
        clean = replace_numbers(clean, lang)
        clean = re.sub(r'\s+', ' ', clean).strip()

        if lang == "hi":
            clean = re.sub(r'[^\u0900-\u097F\s,!?।॥\'"a-z%-]', ' ', clean)
        else:
            clean = re.sub(r'[^a-zA-Z0-9\s,!?.\'"%-]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        if len(clean) < 2: continue
        print(f"  [{i+1}] ({lang}) '{clean[:50]}'")

        out = f"chunk_{i}.wav"
        ok  = False

        try:
            tts.tts_to_file(text=clean, speaker_wav=ref,
                            language=lang, file_path=out, **cfg)
            ok = True
        except Exception as e:
            print(f"  ❌ {e}")
            errors_log.append({"word": clean.split()[0] if clean else "",
                               "error": str(e)})
            try:
                tts.tts_to_file(text=clean, speaker_wav=ref, language=lang,
                                file_path=out, speed=speed,
                                temperature=0.85, repetition_penalty=1.5, top_k=80)
                ok = True
                print(f"  ♻️ Retry OK")
            except Exception as e2:
                print(f"  💀 Retry fail: {e2}")

        if ok and os.path.exists(out):
            seg = AudioSegment.from_wav(out).set_channels(1)
            if use_silence:
                try: seg = effects.strip_silence(seg, silence_thresh=-42, padding=80)
                except: pass
            combined += seg
            pause_ms = get_inter_chunk_pause(chunk)
            if pause_ms > 0:
                combined += AudioSegment.silent(pause_ms)

        if os.path.exists(out): os.remove(out)
        if TORCH_OK and i % 5 == 0:
            torch.cuda.empty_cache(); gc.collect()

    if len(combined) == 0:
        return None, "❌ Koi audio generate nahi hua। Error log check karein।"

    if use_clean:
        combined = match_loudness(combined)
    else:
        combined = combined.set_channels(1).set_frame_rate(44100)

    for f in ["ref_prepared.wav","ref_raw.wav"]:
        if os.path.exists(f): os.remove(f)

    final = "Shri_Ram_Nag_Output.wav"
    combined.export(final, format="wav", parameters=["-ar","44100","-ac","1"])
    print(f"✅ Output ready: {final} ({len(combined)/1000:.1f}s)")

    record_generation(text[:80], total, errors_log)

    if GITHUB_TOKEN and BRAIN_OK:
        try: sync_to_github(GITHUB_TOKEN, GITHUB_REPO)
        except: pass

    msg = f"✅ {total} chunks | {len(combined)/1000:.1f}s"
    if errors_log:
        failed = list(set(e.get("word","") for e in errors_log if e.get("word")))
        msg += f"\n⚠️ {len(errors_log)} error(s) — 'Brain Ko Sikhao' tab mein fix karein"
        if failed: msg += f": {', '.join(failed[:5])}"

    return final, msg

# ════════════════════════════════════════════════════════════
# STARTUP STATUS — App launch hone pe kya ready hai dikhao
# ════════════════════════════════════════════════════════════
def get_system_status():
    lines = ["## 🔧 System Status\n"]
    checks = [
        ("🧠 brain.py", BRAIN_OK),
        ("🔊 TTS Model", tts is not None),
        ("🎵 pydub", PYDUB_OK),
        ("🔢 numpy", NUMPY_OK),
        ("🌐 requests", REQUESTS_OK),
        ("🔥 torch", TORCH_OK),
        ("🤗 huggingface_hub", HF_OK),
        ("🔑 GitHub Token", bool(GITHUB_TOKEN)),
    ]
    all_ok = True
    for name, ok in checks:
        status = "✅" if ok else "❌"
        if not ok: all_ok = False
        lines.append(f"{status} {name}")

    if not all_ok:
        lines.append("\n**Fix karne ke liye:**")
        if not PYDUB_OK: lines.append("```\npip install pydub\n```")
        if not NUMPY_OK: lines.append("```\npip install numpy\n```")
        if tts is None: lines.append("```\npip install TTS\n```")
        if not BRAIN_OK: lines.append("⚠️ brain.py isi folder mein rakho")
    else:
        lines.append("\n✅ **Sab kuch ready hai!**")

    return "\n".join(lines)

# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════
js = """function insertTag(tag) { 
    var t=document.querySelector('#script_box textarea'); 
    if(!t) return;
    var s=t.selectionStart; 
    t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); 
    t.focus(); return t.value; 
}"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js) as demo:

    gr.Markdown("""
    # 🚩 शिव AI v4.1 — श्री राम नाग
    ### Self-Learning | English Fix | Gap Fix | Haklahat-Free
    """)

    # Warning banner agar kuch missing ho
    if not TTS_OK or not PYDUB_OK:
        gr.Markdown("""
        > ⚠️ **Kuch libraries missing hain — niche Status tab mein dekho**
        """)

    with gr.Tabs():

        # ── TAB 1: GENERATION ────────────────────────────
        with gr.TabItem("🎙️ आवाज़ बनाएं"):
            with gr.Row():
                with gr.Column(scale=2):
                    txt = gr.Textbox(
                        label="📝 Script (हिंदी / English / Mixed)",
                        lines=12, elem_id="script_box",
                        placeholder=(
                            "यहाँ script लिखें...\n\n"
                            "उदाहरण:\n"
                            "नमस्कार दोस्तों, आज हम AI technology\n"
                            "के बारे में बात करेंगे।\n"
                            "YouTube पर subscribe करना मत भूलें।"
                        )
                    )
                    wc = gr.Markdown("📊 शब्द: 0")
                    txt.change(
                        lambda x: f"📊 शब्द: **{len(x.split()) if x.strip() else 0}**",
                        [txt],[wc]
                    )
                    with gr.Row():
                        gr.Button("⏸️ [pause]").click(None,None,txt,js="()=>insertTag('[pause]')")
                        gr.Button("💨 [breath]").click(None,None,txt,js="()=>insertTag('[breath]')")
                        gr.Button("😊 [laugh]").click(None,None,txt,js="()=>insertTag('[laugh]')")

                with gr.Column(scale=1):
                    git_ref = gr.Dropdown(
                        choices=["aideva.wav","Joanne.wav"],
                        label="📁 GitHub Voice", value="aideva.wav"
                    )
                    up_ref = gr.Audio(label="🎤 अपनी Voice Upload", type="filepath")
                    with gr.Accordion("⚙️ Settings", open=True):
                        spd = gr.Slider(0.9,1.4,1.1,step=0.05,label="Speed")
                        ptc = gr.Slider(0.8,1.1,0.96,label="Pitch")
                        cln = gr.Checkbox(label="✅ Loudness Match",value=True)
                        sln = gr.Checkbox(label="✅ Silence Remover",value=True)
                    btn = gr.Button("🚀 Generate",variant="primary",size="lg")

            out_audio  = gr.Audio(label="🎧 Output",type="filepath",autoplay=True)
            out_status = gr.Markdown("")

            btn.click(generate,
                      [txt,up_ref,git_ref,spd,ptc,sln,cln],
                      [out_audio,out_status])

        # ── TAB 2: BRAIN TRAINING ─────────────────────────
        with gr.TabItem("🧠 Brain Ko Sikhao"):
            gr.Markdown("""
            ## Brain को नई बात सिखाएं

            कोई English word गलत बोला?  
            नीचे सही बताएं — brain याद रख लेगा।

            | गलत word | सही Hindi phonetic |
            |----------|-------------------|
            | YouTube | यूट्यूब |
            | technology | टेक्नोलॉजी |
            | subscribe | सब्सक्राइब |
            | AI | ए आई |
            """)
            wrong_w   = gr.Textbox(label="❌ गलत word",placeholder="जैसे: technology")
            correct_w = gr.Textbox(label="✅ सही Hindi",placeholder="जैसे: टेक्नोलॉजी")
            teach_btn = gr.Button("🧠 Brain को सिखाओ",variant="primary")
            teach_out = gr.Markdown("")
            teach_btn.click(user_teaches,[wrong_w,correct_w],teach_out)

            gr.Markdown("---")
            stat_btn  = gr.Button("📊 Brain Report देखो")
            brain_out = gr.Markdown("")
            stat_btn.click(get_stats,[],brain_out)

        # ── TAB 3: STATUS ─────────────────────────────────
        with gr.TabItem("🔧 System Status"):
            gr.Markdown(get_system_status())
            gr.Markdown("""
            ---
            ## Requirements Install करने के लिए:
            ```bash
            pip install TTS pydub numpy gradio requests huggingface_hub torch
            ```

            ## brain.py missing है?
            brain.py और app.py **same folder** में होने चाहिए।
            """)

        # ── TAB 4: GITHUB SYNC ────────────────────────────
        with gr.TabItem("🔄 GitHub Sync"):
            gr.Markdown("""
            ## GitHub से Brain Connect करें

            **Hugging Face Spaces पर:**
            ```
            Settings → Variables and Secrets → New Secret
            Name:  GITHUB_TOKEN
            Value: ghp_aapka_token
            ```
            """)
            gh_token = gr.Textbox(label="🔑 GitHub Token",
                                  placeholder="ghp_xxxxxxxx",type="password")
            gh_repo  = gr.Textbox(label="📁 Repo",value="shriramnag/Aivoicebox")
            sync_btn = gr.Button("🔄 Sync Karo",variant="primary")
            sync_out = gr.Markdown("")

            def do_sync(tok,repo):
                if not tok: return "⚠️ Token daalo।"
                return sync_to_github(tok,repo)

            sync_btn.click(do_sync,[gh_token,gh_repo],sync_out)

print("\n✅ App ready — launch ho raha hai...")
demo.launch(share=True)
