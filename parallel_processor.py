import os
from pydub import AudioSegment

def combine_chunks(chunk_files, output_file="shriram_final_pro.wav"):
    if not chunk_files:
        return None

    combined = AudioSegment.empty()
    print(f"🔄 कुल {len(chunk_files)} टुकड़ों को जोड़ा जा रहा है...")

    for file in chunk_files:
        if os.path.exists(file):
            combined += AudioSegment.from_wav(file)
            try:
                os.remove(file) # मेमोरी साफ करना
            except:
                pass
                
    output_path = os.path.abspath(output_file)
    combined.export(output_path, format="wav")
    return output_path
