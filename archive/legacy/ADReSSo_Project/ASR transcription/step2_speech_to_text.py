import os
import glob
import json
from faster_whisper import WhisperModel

# --- CONFIGURATION (Dynamic Relative Paths) ---

# 1. Find the exact folder where this Python script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Set Paths
# INPUT: It reads the audio from Step 1
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed_audios")

# OUTPUT: It saves the JSON transcripts to a new folder
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "transcripts")

# MODEL: Bilgisayarına önceden indirdiğin modeli kullanacak klasör yolu
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "local_whisper_model")

# --- VAD and Acoustic Filters ---
USE_VAD = True
MIN_SILENCE_MS = 2000
SPEECH_PAD_MS = 400
NO_SPEECH_THRESHOLD = 0.4
CONDITION_ON_PREVIOUS_TEXT = False

# Senin özel olarak hazırladığımız "ııı", "şey" dolu prompt'un
FINAL_PROMPT = "Well eee uhm let me ehm let me just aaa look at the iii picture here The the boy is uhm trying to get a aaa cookie out of the jar And ehm the the girl is eee reaching up for it Hmmm tsk the water is is spilling aaa over the sink Eee I I think the iii mother is washing the dishes but ehm she doesn't notice Pfff errr she is is actually aaa stepping in the water The stool is eee uh falling down Wait iii the boy is falling And hmmm ehm what else aaa Ah the the window is open and eee you can see outside Aha okay ehm I think iii that's about everything Uhm eee nineteen seventy two"


def transcribe_audio(file_path, output_path, model):
    """
    Transcribes audio and saves it as a structured JSON with timestamps.
    """
    vad_options = dict(
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=SPEECH_PAD_MS
    ) if USE_VAD else None

    # Run Whisper model
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
    
    # Format the data
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
    
    # Save the JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)


def main():
    """
    Finds the audio files and runs the transcription.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading locally downloaded faster-whisper model...")
    
    # Float/Int ayarlarını kurcalamadığımız ORİJİNAL satır (Senin istediğin gibi)
    model = WhisperModel(LOCAL_MODEL_PATH, device="auto", compute_type="default")

    # Look for both .wav and .mp3 files in the input folder
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    mp3_files = glob.glob(os.path.join(INPUT_DIR, "*.mp3"))
    
    # Combine the lists
    audio_files = wav_files + mp3_files

    if not audio_files:
        print(f"No audio files (.wav or .mp3) found in: {INPUT_DIR}")
        return

    print(f"Found {len(audio_files)} files. Starting transcription...\n")

    for audio_path in audio_files:
        file_name = os.path.basename(audio_path)
        base_name = os.path.splitext(file_name)[0]
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
        
        # Skip if the JSON already exists
        if os.path.exists(json_path):
            print(f"Skipping {file_name} (already processed)")
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