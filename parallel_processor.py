from pydub import AudioSegment
import os
from audio_cleaner import clean_stutter

def combine_chunks(chunk_files, output_file="final_turbo_output.wav"):
    combined = AudioSegment.empty()
    for file in chunk_files:
        if os.path.exists(file):
            clean_stutter(file) # हर टुकड़े से हकलाना हटाएँ [cite: 2026-01-06]
            combined += AudioSegment.from_wav(file)
            os.remove(file)
    combined.export(output_file, format="wav")
    return output_file
