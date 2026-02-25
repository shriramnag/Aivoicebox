"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘      SHIV AI v4.0 â€” à¤¶à¥à¤°à¥€ à¤°à¤¾à¤® à¤¨à¤¾à¤— Voice Cloning             â•‘
â•‘      app.py â€” Main Application                              â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘  SABHI FIXES:                                               â•‘
â•‘  âœ… English words sahi bolega (phonetic Hindi se)            â•‘
â•‘  âœ… Beech mein "dusri line" gap band (0ms inter-chunk)       â•‘
â•‘  âœ… Haklahat nahi (temperature=0.75, rep_penalty=2.5)        â•‘
â•‘  âœ… Self-learning brain â€” galtiyon se seekhta hai            â•‘
â•‘  âœ… GitHub sync â€” restart pe yaad rehta hai                  â•‘
â•‘  âœ… User corrections â€” seedha brain mein jaati hain          â•‘
â•‘  âœ… Mono output, Loudness matched                            â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
"""

import os, torch, gradio as gr, requests, re, gc
import numpy as np
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# â”€â”€ BRAIN IMPORT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from brain import (
    load_memory, load_english_map, save_english_map,
    fix_english_in_hindi, get_inter_chunk_pause,
    record_generation, user_teaches, get_stats,
    sync_to_github, load_from_github
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SETUP
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"ðŸ”§ Device: {device}")

# GitHub Token â€” Hugging Face Spaces mein set karo:
# Settings â†’ Variables and Secrets â†’ GITHUB_TOKEN
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO  = "shriramnag/Aivoicebox"

# App start pe GitHub se brain load karo
if GITHUB_TOKEN:
    print("ðŸ”„ GitHub se brain load ho raha hai...")
    load_from_github(GITHUB_TOKEN, GITHUB_REPO)

# â”€â”€ XTTS MODEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
REPO_ID    = "Shriramnag/My-Shriram-Voice"
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
print(f"âœ… Model: {model_path}")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Custom Ramai.pth weights inject
try:
    ckpt = torch.load(model_path, map_location=device)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict):
        tts.synthesizer.tts_model.load_state_dict(sd, strict=False)
        print("âœ… Ramai.pth weights loaded!")
except Exception as e:
    print(f"âš ï¸  Custom weights skip: {e}")

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# NUMBERS â†’ WORDS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
HINDI_NUMS = {'0':'à¤¶à¥‚à¤¨à¥à¤¯','1':'à¤à¤•','2':'à¤¦à¥‹','3':'à¤¤à¥€à¤¨','4':'à¤šà¤¾à¤°',
              '5':'à¤ªà¤¾à¤à¤š','6':'à¤›à¤¹','7':'à¤¸à¤¾à¤¤','8':'à¤†à¤ ','9':'à¤¨à¥Œ'}
EN_NUMS    = {'0':'zero','1':'one','2':'two','3':'three','4':'four',
              '5':'five','6':'six','7':'seven','8':'eight','9':'nine'}

def replace_numbers(text, lang):
    nmap = HINDI_NUMS if lang == "hi" else EN_NUMS
    def _r(m): return ' '.join(nmap[d] for d in m.group())
    return re.sub(r'\d+', _r, text)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LANGUAGE DETECTION
# Rule: Koi Devanagari = Hindi. Pure English 50%+ chars pe.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def detect_lang(text):
    hi = len(re.findall(r'[\u0900-\u097F]', text))
    en = len(re.findall(r'[a-zA-Z]', text))
    tot = max(len(text.strip()), 1)
    if hi > 0: return "hi"
    if en / tot > 0.5: return "en"
    return "hi"

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SMART TEXT SPLITTER â€” 10 word limit
# XTTS Hindi mein 10+ words ke baad drift karta hai
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
MAX_WORDS = 10

def smart_split(text):
    parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\])', text)
    chunks = []

    for part in parts:
        part = part.strip()
        if not part: continue
        if part in ['[pause]','[breath]','[laugh]']:
            chunks.append(part); continue

        # Sentence boundaries pe split
        sentences = re.split(
            r'(?<=[à¥¤!?à¥¥])\s+|(?<=[.!?])\s+(?=[A-Z\u0900-\u097F])',
            part
        )
        for sent in sentences:
            sent = sent.strip()
            if not sent: continue
            words = sent.split()

            if len(words) <= MAX_WORDS:
                if len(sent) > 1: chunks.append(sent)
            else:
                # Word limit enforce karo
                buf = []
                for w in words:
                    buf.append(w)
                    at_break = (w.endswith((',','â€”','-')) or
                                len(buf) >= MAX_WORDS)
                    if at_break:
                        chunks.append(' '.join(buf)); buf = []
                if buf: chunks.append(' '.join(buf))

    return [c for c in chunks if c]

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# REFERENCE AUDIO PREP
# XTTS speaker encoder: 22050Hz mono prefer karta hai
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def prepare_ref(path):
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(22050)
    audio = effects.normalize(audio)
    if len(audio) < 3000:
        audio = audio * (3000 // len(audio) + 1)
    audio = audio[:30000]
    out = "ref_prepared.wav"
    audio.export(out, format="wav")
    print(f"âœ… Ref ready: {len(audio)/1000:.1f}s")
    return out

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LOUDNESS MATCH â€” Original ke RMS pe laao
# Original RMS: ~4953, Peak: 32393/32767
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def match_loudness(audio, target=4900):
    audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)
    samp = np.array(audio.get_array_of_samples(), dtype=np.float32)
    rms = np.sqrt(np.mean(samp**2))
    if rms > 10:
        gain = min(target/rms, 4.0)
        samp = np.clip(samp*gain, -32767, 32767).astype(np.int16)
        audio = AudioSegment(samp.tobytes(), frame_rate=44100,
                             sample_width=2, channels=1)
    return effects.normalize(audio)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TTS PARAMS â€” Haklahat fix
# temperature=0.75: freeze nahi hoga
# rep_penalty=2.5: real repetition hi rokega
# top_k=50: enough Hindi vocab
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def tts_cfg(speed):
    return dict(temperature=0.75, repetition_penalty=2.5,
                top_k=50, top_p=0.85, speed=speed)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN GENERATE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def generate(text, up_ref, git_ref, speed, pitch,
             use_silence, use_clean, progress=gr.Progress()):

    emap = load_english_map()  # Brain se English map load
    errors_log = []
    lang_log = []

    # Reference prepare
    if up_ref:
        ref = prepare_ref(up_ref)
    else:
        raw = "ref_raw.wav"
        url = G_RAW + requests.utils.quote(git_ref)
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None, f"âŒ GitHub se voice nahi mili ({r.status_code})"
        with open(raw,"wb") as f: f.write(r.content)
        ref = prepare_ref(raw)

    chunks = smart_split(text)
    total  = len(chunks)
    print(f"ðŸ“ {total} chunks")

    combined = AudioSegment.empty()
    cfg = tts_cfg(speed)

    for i, chunk in enumerate(chunks):
        progress((i+1)/total, desc=f"ðŸŽ™ï¸ {i+1}/{total}: {chunk[:30]}...")

        # Markers
        if chunk == "[pause]":
            combined += AudioSegment.silent(800); continue
        elif chunk == "[breath]":
            combined += AudioSegment.silent(300); continue
        elif chunk == "[laugh]":
            combined += AudioSegment.silent(100); continue

        lang = detect_lang(chunk)
        lang_log.append({"chunk": chunk[:40], "lang": lang})

        # â”€â”€ ENGLISH FIX (Brain) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Hindi text mein English words â†’ phonetic Hindi
        if lang == "hi":
            clean = fix_english_in_hindi(chunk, emap)
        else:
            clean = chunk

        clean = replace_numbers(clean, lang)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Safe chars
        if lang == "hi":
            clean = re.sub(r'[^\u0900-\u097F\s,!?à¥¤à¥¥\'"a-z%-]', ' ', clean)
        else:
            clean = re.sub(r'[^a-zA-Z0-9\s,!?.\'"%-]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        if len(clean) < 2: continue

        print(f"  [{i+1}] ({lang}) '{clean[:50]}'")
        out = f"chunk_{i}.wav"
        ok = False

        try:
            tts.tts_to_file(text=clean, speaker_wav=ref,
                            language=lang, file_path=out, **cfg)
            ok = True
        except Exception as e:
            print(f"  âŒ Fail: {e}")
            errors_log.append({
                "chunk": clean[:40], "lang": lang,
                "error": str(e),
                "word": clean.split()[0] if clean else ""
            })
            # Retry â€” loose params
            try:
                tts.tts_to_file(text=clean, speaker_wav=ref,
                                language=lang, file_path=out,
                                speed=speed, temperature=0.85,
                                repetition_penalty=1.5, top_k=80)
                ok = True
                print(f"  â™»ï¸  Retry success")
            except Exception as e2:
                print(f"  ðŸ’€ Retry fail: {e2}")

        if ok and os.path.exists(out):
            seg = AudioSegment.from_wav(out).set_channels(1)
            if use_silence:
                try: seg = effects.strip_silence(seg, silence_thresh=-42, padding=80)
                except: pass
            combined += seg

            # â”€â”€ GAP FIX â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Har chunk ke baad content-aware pause
            # (Pehle sab jagah 60ms tha â€” yahi "dusri line" thi)
            pause_ms = get_inter_chunk_pause(chunk)
            if pause_ms > 0:
                combined += AudioSegment.silent(pause_ms)

        if os.path.exists(out): os.remove(out)

        if i % 5 == 0:
            torch.cuda.empty_cache(); gc.collect()

    # Final audio
    if use_clean:
        combined = match_loudness(combined)
    else:
        combined = combined.set_channels(1).set_frame_rate(44100)

    for f in ["ref_prepared.wav","ref_raw.wav"]:
        if os.path.exists(f): os.remove(f)

    final = "Shri_Ram_Nag_Output.wav"
    combined.export(final, format="wav", parameters=["-ar","44100","-ac","1"])
    print(f"âœ… Output: {final} ({len(combined)/1000:.1f}s)")

    # â”€â”€ BRAIN SEEKHTA HAI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    record_generation(text[:80], total, errors_log, 0)

    # GitHub sync (har generation ke baad)
    if GITHUB_TOKEN:
        sync_result = sync_to_github(GITHUB_TOKEN, GITHUB_REPO)
        print(f"ðŸ”„ {sync_result}")

    status_msg = f"âœ… Taiyaar! {total} chunks"
    if errors_log:
        status_msg += f" | âš ï¸ {len(errors_log)} error(s) â€” brain ne yaad rakha"
        # Brain ne failre yaad rakha â€” agli baar fix try karega
        failed_words = [e.get("word","") for e in errors_log if e.get("word")]
        if failed_words:
            status_msg += f"\nðŸ’¡ Sikhane ke liye 'Brain Ko Sikhao' tab mein jaayein: {', '.join(set(failed_words))}"

    return final, status_msg

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# UI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
js = """function insertTag(tag) { 
    var t=document.querySelector('#script_box textarea'); 
    var s=t.selectionStart; 
    t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); 
    t.focus(); return t.value; 
}"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js) as demo:

    gr.Markdown("""
    # ðŸš© à¤¶à¤¿à¤µ AI v4.0 â€” à¤¶à¥à¤°à¥€ à¤°à¤¾à¤® à¤¨à¤¾à¤—
    ### Self-Learning | English Fix | Gap Fix | Haklahat-Free
    > âœ… English phonetic fix &nbsp;|&nbsp; âœ… Gap fix (0ms) &nbsp;|&nbsp; 
    > âœ… temperature=0.75 &nbsp;|&nbsp; âœ… GitHub brain sync
    """)

    with gr.Tabs():

        # â”€â”€ TAB 1: GENERATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with gr.TabItem("ðŸŽ™ï¸ à¤†à¤µà¤¾à¤œà¤¼ à¤¬à¤¨à¤¾à¤à¤‚"):
            with gr.Row():
                with gr.Column(scale=2):
                    txt = gr.Textbox(
                        label="ðŸ“ à¤¸à¥à¤•à¥à¤°à¤¿à¤ªà¥à¤Ÿ (à¤¹à¤¿à¤‚à¤¦à¥€ / English / Mixed)",
                        lines=12, elem_id="script_box",
                        placeholder=(
                            "à¤¯à¤¹à¤¾à¤ script à¤²à¤¿à¤–à¥‡à¤‚...\n\n"
                            "à¤‰à¤¦à¤¾à¤¹à¤°à¤£:\n"
                            "à¤¨à¤®à¤¸à¥à¤•à¤¾à¤° à¤¦à¥‹à¤¸à¥à¤¤à¥‹à¤‚, à¤†à¤œ à¤¹à¤® AI technology à¤•à¥‡ à¤¬à¤¾à¤°à¥‡ à¤®à¥‡à¤‚ à¤¬à¤¾à¤¤ à¤•à¤°à¥‡à¤‚à¤—à¥‡à¥¤\n"
                            "YouTube à¤ªà¤° subscribe à¤•à¤°à¤¨à¤¾ à¤®à¤¤ à¤­à¥‚à¤²à¥‡à¤‚à¥¤\n\n"
                            "ðŸ’¡ English words à¤…à¤ªà¤¨à¥‡ à¤†à¤ª à¤ à¥€à¤• à¤¹à¥‹ à¤œà¤¾à¤à¤‚à¤—à¥‡à¥¤\n"
                            "ðŸ’¡ [pause] à¤¸à¥‡ à¤°à¥à¤•à¤¾à¤µà¤Ÿ, [breath] à¤¸à¥‡ à¤¸à¤¾à¤‚à¤¸ à¤œà¥‹à¤¡à¤¼à¥‡à¤‚à¥¤"
                        )
                    )
                    wc = gr.Markdown("ðŸ“Š à¤¶à¤¬à¥à¤¦: 0")
                    txt.change(
                        lambda x: f"ðŸ“Š à¤¶à¤¬à¥à¤¦: **{len(x.split()) if x.strip() else 0}**",
                        [txt],[wc]
                    )
                    with gr.Row():
                        gr.Button("â¸ï¸ [pause]").click(None,None,txt,js="()=>insertTag('[pause]')")
                        gr.Button("ðŸ’¨ [breath]").click(None,None,txt,js="()=>insertTag('[breath]')")
                        gr.Button("ðŸ˜Š [laugh]").click(None,None,txt,js="()=>insertTag('[laugh]')")

                with gr.Column(scale=1):
                    git_ref = gr.Dropdown(
                        choices=["aideva.wav","Joanne.wav"],
                        label="ðŸ“ GitHub Voice", value="aideva.wav"
                    )
                    up_ref = gr.Audio(label="ðŸŽ¤ à¤…à¤ªà¤¨à¥€ Voice Upload", type="filepath")
                    with gr.Accordion("âš™ï¸ Settings", open=True):
                        spd = gr.Slider(0.9,1.4,1.1,step=0.05,label="Speed (1.1 = best)")
                        ptc = gr.Slider(0.8,1.1,0.96,label="Pitch")
                        cln = gr.Checkbox(label="âœ… Loudness Match", value=True)
                        sln = gr.Checkbox(label="âœ… Silence Remover", value=True)
                    btn = gr.Button("ðŸš€ Generate", variant="primary", size="lg")

            out_audio  = gr.Audio(label="ðŸŽ§ Output", type="filepath", autoplay=True)
            out_status = gr.Markdown("")

            btn.click(generate,
                      [txt,up_ref,git_ref,spd,ptc,sln,cln],
                      [out_audio, out_status])

        # â”€â”€ TAB 2: BRAIN TRAINING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with gr.TabItem("ðŸ§  Brain Ko Sikhao"):
            gr.Markdown("""
            ## Brain Ko à¤¨à¤ˆ à¤¬à¤¾à¤¤ à¤¸à¤¿à¤–à¤¾à¤à¤‚

            à¤…à¤—à¤° à¤•à¥‹à¤ˆ **English word à¤—à¤²à¤¤ à¤¬à¥‹à¤²à¤¾** à¤œà¤¾ à¤°à¤¹à¤¾ à¤¹à¥ˆ â€”  
            à¤¨à¥€à¤šà¥‡ à¤¸à¤¹à¥€ à¤¬à¤¤à¤¾à¤à¤‚à¥¤ Brain à¤¯à¤¾à¤¦ à¤°à¤– à¤²à¥‡à¤—à¤¾à¥¤

            **à¤‰à¤¦à¤¾à¤¹à¤°à¤£ corrections:**
            | à¤—à¤²à¤¤ word | à¤¸à¤¹à¥€ Hindi |
            |----------|-----------|
            | YouTube | à¤¯à¥‚à¤Ÿà¥à¤¯à¥‚à¤¬ |
            | subscribe | à¤¸à¤¬à¥à¤¸à¤•à¥à¤°à¤¾à¤‡à¤¬ |
            | AI | à¤ à¤†à¤ˆ |
            | technology | à¤Ÿà¥‡à¤•à¥à¤¨à¥‹à¤²à¥‰à¤œà¥€ |
            """)

            with gr.Row():
                with gr.Column():
                    wrong_w   = gr.Textbox(label="âŒ à¤•à¥Œà¤¨ à¤¸à¤¾ word à¤—à¤²à¤¤ à¤¬à¥‹à¤²à¤¾?",
                                           placeholder="à¤œà¥ˆà¤¸à¥‡: technology")
                    correct_w = gr.Textbox(label="âœ… à¤¸à¤¹à¥€ Hindi phonetic?",
                                           placeholder="à¤œà¥ˆà¤¸à¥‡: à¤Ÿà¥‡à¤•à¥à¤¨à¥‹à¤²à¥‰à¤œà¥€")
                    teach_btn = gr.Button("ðŸ§  Brain à¤•à¥‹ à¤¸à¤¿à¤–à¤¾à¤“", variant="primary")
                    teach_out = gr.Markdown("")
                    teach_btn.click(user_teaches, [wrong_w, correct_w], teach_out)

            gr.Markdown("---")
            gr.Markdown("## ðŸ“Š Brain Report â€” à¤…à¤¬ à¤¤à¤• à¤•à¥à¤¯à¤¾ à¤¸à¥€à¤–à¤¾?")
            stat_btn    = gr.Button("ðŸ” Report à¤¦à¥‡à¤–à¥‹")
            brain_stats = gr.Markdown("")
            stat_btn.click(get_stats, [], brain_stats)

        # â”€â”€ TAB 3: GITHUB SYNC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with gr.TabItem("ðŸ”„ GitHub Brain Sync"):
            gr.Markdown("""
            ## GitHub à¤¸à¥‡ Brain Connect à¤•à¤°à¥‡à¤‚

            **à¤•à¥à¤¯à¥‹à¤‚ à¤œà¤°à¥‚à¤°à¥€:**  
            Server restart à¤¹à¥‹à¤¨à¥‡ à¤ªà¤° brain à¤•à¥€ memory à¤–à¥‹ à¤œà¤¾à¤¤à¥€ à¤¹à¥ˆà¥¤  
            GitHub token à¤¦à¥‡à¤¨à¥‡ à¤ªà¤° memory save à¤¹à¥‹à¤¤à¥€ à¤°à¤¹à¤¤à¥€ à¤¹à¥ˆ â€”  
            à¤…à¤—à¤²à¥€ à¤¬à¤¾à¤° app start à¤¹à¥‹ à¤¤à¥‹ à¤¸à¤¬ à¤¯à¤¾à¤¦ à¤°à¤¹à¤¤à¤¾ à¤¹à¥ˆà¥¤

            **Hugging Face Spaces à¤ªà¤° token à¤•à¥ˆà¤¸à¥‡ set à¤•à¤°à¥‡à¤‚:**
            ```
            Settings â†’ Variables and Secrets â†’ New Secret
            Name: GITHUB_TOKEN
            Value: ghp_aapka_token_yahan
            ```

            **GitHub Token à¤•à¥ˆà¤¸à¥‡ à¤¬à¤¨à¤¾à¤à¤‚:**
            ```
            GitHub â†’ Settings â†’ Developer settings
            â†’ Personal access tokens â†’ Tokens (classic)
            â†’ Generate â†’ repo permission à¤¦à¥‡à¤‚ â†’ Copy
            ```
            """)

            with gr.Row():
                gh_token = gr.Textbox(label="ðŸ”‘ GitHub Token",
                                      placeholder="ghp_xxxxxxxxxxxxxxxx",
                                      type="password")
                gh_repo  = gr.Textbox(label="ðŸ“ Repo",
                                      value="shriramnag/Aivoicebox")

            sync_btn = gr.Button("ðŸ”„ à¤…à¤­à¥€ Sync à¤•à¤°à¥‹", variant="primary")
            sync_out = gr.Markdown("")

            def manual_sync(tok, repo):
                if not tok: return "âš ï¸ Token daaloà¥¤"
                return sync_to_github(tok, repo)

            sync_btn.click(manual_sync, [gh_token, gh_repo], sync_out)

demo.launch(share=True)
