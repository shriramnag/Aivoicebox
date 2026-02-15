from pydub import AudioSegment
import os
from audio_cleaner import clean_audio

def combine_chunks(chunk_files, output_file="shriram_final_pro.wav"):
    combined = AudioSegment.empty()
    for file in chunk_files:
        if os.path.exists(file):
            clean_audio(file) # हर टुकड़े की सफाई [cite: 2026-01-06]
            combined += AudioSegment.from_wav(file)
            os.remove(file)
    combined.export(output_file, format="wav")
    return output_file
