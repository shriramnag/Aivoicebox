from pydub import AudioSegment
import os

def combine_audio_chunks(chunk_files, output_filename="final_audio.wav"):
    # प्रोफेशनल तरीके से ऑडियो को जोड़ना [cite: 2026-01-06]
    combined = AudioSegment.empty()
    for file in chunk_files:
        if os.path.exists(file):
            combined += AudioSegment.from_wav(file)
            os.remove(file) # टेम्पररी फाइल डिलीट करना
    combined.export(output_filename, format="wav")
    return output_filename
  
