import os
import sys
import torch
import numpy as np
import ffmpeg
import whisper
from pydub import AudioSegment
from transformers import pipeline
from gtts import gTTS
from rouge_score import rouge_scorer
import math
import time

def check_dependencies():
    try:
        _ = torch.__version__
        _ = np.__version__
        _ = ffmpeg.__version__ if hasattr(ffmpeg, "__version__") else "ffmpeg installed"
        _ = whisper.load_model("tiny")
        _ = AudioSegment.converter
        _ = pipeline
        _ = gTTS
        _ = rouge_scorer.RougeScorer
        print("All dependencies are installed and accessible!")
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        return False

device = "cuda" if torch.cuda.is_available() else "cpu"

### Step 1: Extract Audio ###
def extract_audio(video_path: str, audio_path: str = "audio.wav"):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print("Extracting audio from video...")
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(audio_path, format="wav")
        print("Audio extracted and saved as audio.wav")
        return audio_path
    except Exception as e:
        print(f"Audio extraction error: {str(e)}")
        return None

### Step 2: Transcribe in Chunks ###
def transcribe_audio_chunkwise(audio_path: str, model_size="tiny", chunk_length_ms=30*1000):
    try:
        model = whisper.load_model(model_size).to("cpu") 
        audio = AudioSegment.from_wav(audio_path)
        transcript = ""
        total_chunks = math.ceil(len(audio) / chunk_length_ms)
        print(f"Starting transcription in {total_chunks} chunks...")

        for i in range(total_chunks):
            start = i * chunk_length_ms
            end = min((i + 1) * chunk_length_ms, len(audio))
            chunk = audio[start:end]
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")

            result = model.transcribe(chunk_path, fp16=False, language="en")
            transcript += result["text"] + " "

            os.remove(chunk_path)
            print(f"Transcribed chunk {i+1}/{total_chunks} ({round((i+1)/total_chunks*100, 2)}%)")

        return transcript.strip()
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None

### Step 3: Summarize Text ###
def summarize_text(text: str):
    try:
        print("Starting summarization...")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if device=="cuda" else -1)
        max_chunk_len = 800
        chunks = [text[i:i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]
        summaries = []

        for idx, chunk in enumerate(chunks):
            print(f"Summarizing chunk {idx+1}/{len(chunks)} ({round((idx+1)/len(chunks)*100, 2)}%)...")
            summary = summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
            time.sleep(0.5) 

        return " ".join(summaries)
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return None

### Step 3.5: Evaluate Summary Accuracy ###
def evaluate_summary_accuracy(summary: str, reference_text: str):
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, summary)

        print("\nAccuracy (ROUGE Scores):")
        for key, score in scores.items():
            print(f"🔹 {key}: Precision={score.precision:.2f}, Recall={score.recall:.2f}, F1={score.fmeasure:.2f}")
        
        return scores["rouge1"].fmeasure
    except Exception as e:
        print(f"Accuracy evaluation error: {str(e)}")
        return None

### Step 4: Convert Summary to Audio ###
def text_to_speech(summary_text: str, output_audio: str = "summary.mp3"):
    try:
        print("Converting text summary to audio...")
        tts = gTTS(summary_text)
        tts.save(output_audio)
        print(f"Summary saved as: {output_audio}")
        return output_audio
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        return None

### Step 5: Clean Temporary Files ###
def cleanup_temp_files(audio_path: str):
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("Temporary files cleaned up.")
    except Exception as e:
        print(f"Cleanup warning: {str(e)}")

### Step 6: Pipeline Execution ###
def process_video(video_path: str):
    if not check_dependencies():
        return False

    print("Starting full processing pipeline...\n")

    audio_path = extract_audio(video_path)
    if not audio_path:
        return False

    transcript = transcribe_audio_chunkwise(audio_path)
    if not transcript:
        cleanup_temp_files(audio_path)
        return False
    
    print("\nTranscript (first 300 chars):\n", transcript[:300], "...\n")

    summary = summarize_text(transcript)
    if not summary:
        cleanup_temp_files(audio_path)
        return False

    print("\nFinal Summary:\n", summary, "\n")

    # 🔍 Evaluate accuracy
    accuracy_score = evaluate_summary_accuracy(summary, transcript)
    if accuracy_score is not None:
        print(f"\nEstimated Summary Accuracy (ROUGE-1 F1): {accuracy_score*100:.2f}%")

    summary_audio = text_to_speech(summary)
    if not summary_audio:
        cleanup_temp_files(audio_path)
        return False

    cleanup_temp_files(audio_path)
    print("Process completed successfully!")
    return True

# Entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Main_Project.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"File does not exist: {video_path}")
        sys.exit(1)

    success = process_video(video_path)
    sys.exit(0 if success else 1)
