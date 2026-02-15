import os
from pydub import AudioSegment

def combine_chunks(chunk_files, output_file="shriram_final_pro.wav"):
    """सभी ऑडियो टुकड़ों को बिना किसी शोर के जोड़ना"""
    if not chunk_files:
        return None

    combined = AudioSegment.empty()
    print(f"🔄 कुल {len(chunk_files)} टुकड़ों को जोड़ा जा रहा है...")

    for file in chunk_files:
        if os.path.exists(file):
            try:
                segment = AudioSegment.from_wav(file)
                combined += segment
                # पुराने टुकड़ों को हटाना ताकि मेमोरी फुल न हो [cite: 2026-01-06]
                os.remove(file) 
            except Exception as e:
                print(f"Error processing {file}: {e}")
                
    output_path = os.path.abspath(output_file)
    combined.export(output_path, format="wav")
    print(f"✅ फाइनल फाइल तैयार: {output_path}")
    return output_path
