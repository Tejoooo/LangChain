import requests
import json
import os
import sounddevice as sd
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from elevenlabs import ElevenLabs,VoiceSettings
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import google.generativeai as genai
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY')
ELEVEN_LABS_API_KEY = "sk_06c7f34d0ad0c692feafecb34058ac45d61ca5ac839d6fce"

client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)



def record_audio(filename="input_audio.wav", duration=5, samplerate=44100):
    print("Recording... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, samplerate, audio_data)
    print("Recording saved as input_audio.wav")
    return filename


def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        response = client.speech_to_text.convert(
        model_id="scribe_v1",  
        file=audio_file  
    )

    return response.text


def get_gemini_response(text):
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {"temperature" :0.9,"top_p":1,"top_k":1,"max_output_tokens":100}
    model = genai.GenerativeModel("gemini-1.5-pro",generation_config=generation_config)
    
    response = model.generate_content(text)
    print(response.text)
      
    
    try:
        return response.text
    except requests.exceptions.JSONDecodeError:
        print("Error: Response is not in JSON format.")
        return ""



def generate_speech(text, output_file="response.mp3"):
    response = client.text_to_speech.convert(
        model_id="eleven_multilingual_v2",
        voice_id="pNInz6obpgDQGcFmaJgB",
        output_format="mp3_22050_32",
        text=text,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=1.0,
            style=0.5,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    with open(output_file, "wb") as audio_file:
        for chunk in response:
            if chunk:
                audio_file.write(chunk)

    print(f"Audio saved as {output_file}")


def play_speech(text):
    response = client.text_to_speech.convert(
        model_id="eleven_multilingual_v2",
        voice_id="pNInz6obpgDQGcFmaJgB",
        output_format="pcm_16000",
        text=text,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=1.0,
            style=0.5,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    audio_stream = BytesIO()

    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    audio_stream.seek(0)
    
    # Convert byte stream to numpy array for playback
    audio_data = np.frombuffer(audio_stream.read(), dtype=np.int16)

    # Play audio using sounddevice
    sd.play(audio_data, samplerate=16000)
    sd.wait()

if __name__ == "__main__":
    record_audio = record_audio()
    audio_text = transcribe_audio(record_audio)  
    print(f"Transcribed text: {audio_text}")

    gemini_reply = get_gemini_response(audio_text) 
    print(f"Gemini Response: {gemini_reply}")

    play_speech(gemini_reply)  
