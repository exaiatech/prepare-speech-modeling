import os
import json
import logging
from pathlib import Path
from faster_whisper import WhisperModel

# 1. Enable logging to see the download progress of the large-v3 model
logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parents[2] / "data" / "interim" / "processed_audios"
OUTPUT_DIR = SCRIPT_DIR.parents[2] / "data" / "interim" / "transcripts"

LOCAL_MODEL_PATH = SCRIPT_DIR.parents[2] / "models" / "large-v3-local"

# --- VAD AND ACOUSTIC FILTERS ---
USE_VAD = True
MIN_SILENCE_MS = 2000
SPEECH_PAD_MS = 400
NO_SPEECH_THRESHOLD = 0.4
CONDITION_ON_PREVIOUS_TEXT = True

FINAL_PROMPT = "Well eee uhm let me ehm let me just aaa look at the iii picture here The the boy is uhm trying to get a aaa cookie out of the jar And ehm the the girl is eee reaching up for it Hmmm tsk the water is is spilling aaa over the sink Eee I I think the iii mother is washing the dishes but ehm she doesn't notice Pfff errr she is is actually aaa stepping in the water The stool is eee uh falling down Wait iii the boy is falling And hmmm ehm what else aaa Ah the the window is open and eee you can see outside Aha okay ehm I think iii that's about everything Uhm eee nineteen seventy two"

def transcribe_audio(file_path, output_path, model):
    """
    Transcribes audio and saves it as a structured JSON.
    """
    vad_options = dict(
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=SPEECH_PAD_MS
    ) if USE_VAD else None

    # Run Whisper model (converting Path object to string)
    segments, _ = model.transcribe(
        str(file_path),
        language="en",
        initial_prompt=FINAL_PROMPT,
        word_timestamps=True,
        vad_filter=USE_VAD,
        vad_parameters=vad_options,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
    )
    
    extracted_data = {
        "file_name": file_path.name,
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
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # The checkpoint is 2.9 GB and is not in the repository. Without this
    # check faster-whisper reports "Invalid model size <path>", which does not
    # tell you what is actually wrong.
    if not (LOCAL_MODEL_PATH / "model.bin").exists():
        print(f"ERROR: no Whisper checkpoint at {LOCAL_MODEL_PATH}")
        print("       Download it first. See models/README.md.")
        return

    # Update this line to load from your local path instead of internet
    print(f"Loading local Whisper model from: {LOCAL_MODEL_PATH}")
    model = WhisperModel(str(LOCAL_MODEL_PATH), device="auto", compute_type="default")


    # Find all .wav and .mp3 files cleanly using rglob
    audio_files = list(INPUT_DIR.rglob("*.wav")) + list(INPUT_DIR.rglob("*.mp3"))

    if not audio_files:
        print(f"No audio files found in: {INPUT_DIR}")
        return

    print(f"\nFound {len(audio_files)} files. Starting transcription...\n")

    for audio_path in audio_files:
        output_file_name = f"{audio_path.stem}.json"
        json_path = OUTPUT_DIR / output_file_name
        
        if json_path.exists():
            print(f"Skipping {audio_path.name} (already processed)")
            continue
            
        print(f"Processing: {audio_path.name}")
        
        try:
            transcribe_audio(audio_path, json_path, model)
            print(f" -> Saved: {output_file_name}")
        except Exception as e:
            print(f" -> ERROR processing {audio_path.name}: {e}")

    print("\nAll files processed successfully.")

if __name__ == "__main__":
    main()