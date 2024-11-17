from transformers import pipeline, VitsModel, AutoTokenizer  # Import pipeline, VITS model, and tokenizer from Hugging Face Transformers
import torch  # Import PyTorch for tensor operations and model inference
import torchaudio  # Import torchaudio for saving audio files

# Convert audio file to text, translate the text to Tok Pisin, then convert back to speech
def process_audio(input_audio_path, output_audio_path):
    # Initialize the MMS pipeline for automatic speech recognition
    # This pipeline converts speech in an audio file to text
    asr_pipe = pipeline("automatic-speech-recognition", model="facebook/mms-1b-all")
    
    # Perform speech recognition on the input audio file
    # The pipeline takes the audio file path and returns the recognized text
    result = asr_pipe(input_audio_path)
    transcribed_text = result['text']  # Extract the transcribed text from the result
    
    print(f"Transcribed text: {transcribed_text}")  # Print the transcribed text to see the result
    
    # Initialize the translation pipeline for translating from English to Tok Pisin
    # Make sure you have installed `sentencepiece` for this tokenizer to work properly
    translation_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-tpi")
    
    # Translate the transcribed text to Tok Pisin
    translated_text = translation_pipe(transcribed_text)[0]['translation_text']
    
    print(f"Translated text: {translated_text}")  # Print the translated text to see the result
    
    # Initialize the VITS model and tokenizer for text-to-speech conversion in Tok Pisin
    model = VitsModel.from_pretrained("facebook/mms-tts-tpi")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tpi")
    
    # Tokenize the translated text
    # The tokenizer converts the text into a format that the model can process
    inputs = tokenizer(translated_text, return_tensors="pt")  # "pt" means PyTorch tensor format
    
    # Generate audio waveform from the translated text using VITS model
    # The model takes the tokenized input and produces the audio waveform
    with torch.no_grad():  # Disable gradient calculation since we are only doing inference
        output = model(**inputs).waveform  # Generate waveform from the model
    
    # Ensure the output is 2D (channels x samples) before saving
    if output.dim() == 1:
        output = output.unsqueeze(0)  # Add a channel dimension if the waveform is 1D
    
    # Save the generated audio to output file using torchaudio
    # Use torchaudio to save the waveform to an audio file (WAV format)
    torchaudio.save(output_audio_path, output, sample_rate=model.config.sampling_rate)
    
    print(f"Audio file saved to: {output_audio_path}")  # Print confirmation that the audio file is saved

# Main function to run the entire process
def main(input_audio_path, output_audio_path):
    # Call the process_audio function to convert speech to text, translate, and convert back to speech
    process_audio(input_audio_path, output_audio_path)

# Entry point of the script
if __name__ == "__main__":
    # Input audio file path (change this to the location of your audio file)
    input_audio_path = "./test-input.wav"
    # Output audio file path (change this to the desired location and name of the output audio file)
    output_audio_path = "./test-output.wav"
    
    # Call the main function with the input and output audio file paths
    main(input_audio_path, output_audio_path)
    
    # Print completion message
    print("Speech to text to translation to speech conversion completed.")
