import os
import glob
import json
from faster_whisper import WhisperModel

# --- CONFIGURATION (Final Methodology Settings) ---
INPUT_DIR = r"C:\Users\arday\Desktop\UROP\TEST\Second\test_voice" 
OUTPUT_DIR = r"C:\Users\arday\Desktop\UROP\TEST\Second\test_transcript\Final_Pipeline_Output"
LOCAL_MODEL_PATH = r"C:\Users\arday\Desktop\local_whisper_model"

# VAD and Acoustic Filters (Based on experimental baseline)
USE_VAD = True
MIN_SILENCE_MS = 2000
SPEECH_PAD_MS = 400
NO_SPEECH_THRESHOLD = 0.4
CONDITION_ON_PREVIOUS_TEXT = False

# New Simple Prompt
FINAL_PROMPT = "Well uh let me look at the picture here. The boy is trying to get a cookie out of the jar. And um the girl is reaching up for it. Hmmm the water is spilling over the sink. Eee I think the mother is washing the dishes but she doesn't notice. Haha she is is actually stepping in the water. The stool is uh falling down. W-wait the boy is falling. And hm what else. Ah the window is open and you can see outside. Aha okay I think that's about everything. Um nineteen seventy two."


def transcribe_audio(file_path, output_path, model):
    """
    Transcribes a single audio file and saves the result as a structured JSON.
    Includes word-level timestamps.
    """
    vad_options = dict(
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=SPEECH_PAD_MS
    ) if USE_VAD else None

    # Run Whisper inference
    segments, _ = model.transcribe(
        file_path,
        language="en",
        initial_prompt=FINAL_PROMPT,
        word_timestamps=True,
        vad_filter=USE_VAD,
        vad_parameters=vad_options,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
    )
    
    # Format the data structure
    extracted_data = {
        "file_name": os.path.basename(file_path),
        "full_text": "",
        "words": []
    }
    
    full_text_parts = []
    for segment in segments:
        full_text_parts.append(segment.text.strip())
        for word in segment.words:
            extracted_data["words"].append({
                "word": word.word.strip(),
                "start": round(word.start, 3),
                "end": round(word.end, 3)
            })
            
    extracted_data["full_text"] = " ".join(full_text_parts)
    
    # Save the structured JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)


def main():
    """
    Main execution loop to process all audio files.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading faster-whisper model...")
    model = WhisperModel(LOCAL_MODEL_PATH, device="auto", compute_type="default")

    audio_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    print(f"Found {len(audio_files)} files. Starting final transcription pipeline...\n")

    for audio_path in audio_files:
        file_name = os.path.basename(audio_path)
        base_name = os.path.splitext(file_name)[0]
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
        
        # Skip if already processed
        if os.path.exists(json_path):
            continue
            
        print(f"Processing: {file_name}")
        
        try:
            transcribe_audio(audio_path, json_path, model)
            print(f" -> Saved: {base_name}.json")
        except Exception as e:
            print(f" -> ERROR processing {file_name}: {e}")

    print("\nAll files processed successfully.")


if __name__ == "__main__":
    main()