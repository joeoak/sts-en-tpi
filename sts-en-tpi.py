import argparse
import torchaudio
from transformers import pipeline, VitsModel, AutoTokenizer

# Convert audio file to text, translate text to Tok Pisin, then convert translated text to speech
def sts_en_tpi(input_audio_path, output_audio_path):
    # Initialize speech recognition
    asr_pipe = pipeline("automatic-speech-recognition", model="facebook/mms-1b-all")
    
    # Recognize speech from audio file
    result = asr_pipe(input_audio_path)
    transcribed_text = result['text']  # Extract text
    
    print(f"Transcribed text: {transcribed_text}")  # Print transcribed text
    
    # Initialize translation pipeline to translate text to Tok Pisin
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-tpi")
    
    # Translate transcribed text to Tok Pisin
    translated_text = translator(transcribed_text)[0]['translation_text']  # Extract translated text
    
    print(f"Translated text: {translated_text}")  # Print translated text
    
    # Initialize VITS model and tokenizer for text-to-speech conversion
    model = VitsModel.from_pretrained("facebook/mms-tts-tpi")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tpi")
    
    # Tokenize translated text
    input_ids = tokenizer(translated_text, return_tensors="pt").input_ids
    
    # Convert translated text to speech
    audio = model(input_ids).waveform  # Generate audio from text
    
    # Save generated audio to file
    torchaudio.save(output_audio_path, audio, 16000)  # Save audio at 16kHz
    
    print(f"Audio saved at: {output_audio_path}")  # Print confirmation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio file from English to Tok Pisin")
    parser.add_argument("input_audio_path", type=str, help="Path to input audio file")
    parser.add_argument("output_audio_path", type=str, help="Path to save output audio file")
    args = parser.parse_args()

    sts_en_tpi(args.input_audio_path, args.output_audio_path)
