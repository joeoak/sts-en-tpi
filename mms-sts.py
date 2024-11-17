# Import necessary libraries
from transformers import pipeline, VitsModel, AutoTokenizer  # Import pipeline, VITS model, and tokenizer from Hugging Face Transformers
import torch  # Import PyTorch for tensor operations and model inference
import torchaudio  # Import torchaudio for saving audio files

# Convert audio file to text, then convert back to speech using MMS pipeline
def process_audio_with_mms(input_audio_path, output_audio_path):
    # Initialize the MMS pipeline for automatic speech recognition
    # This pipeline converts speech in an audio file to text
    asr_pipe = pipeline("automatic-speech-recognition", model="facebook/mms-1b-all")
    
    # Perform speech recognition on the input audio file
    # The pipeline takes the audio file path and returns the recognized text
    result = asr_pipe(input_audio_path)
    transcribed_text = result['text']  # Extract the transcribed text from the result
    
    print(f"Transcribed text: {transcribed_text}")  # Print the transcribed text to see the result
    
    # Initialize the VITS model and tokenizer for text-to-speech conversion
    # The VITS model generates speech from the given text
    model = VitsModel.from_pretrained("facebook/mms-tts-spa")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")
    
    # Tokenize the transcribed text
    # The tokenizer converts the text into a format that the model can process
    inputs = tokenizer(transcribed_text, return_tensors="pt")  # "pt" means PyTorch tensor format
    
    # Generate audio waveform from the transcribed text using VITS model
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

# End to end conversion (audio file -> audio file)
def main(input_audio_path, output_audio_path):
    # Convert the audio file using MMS pipeline
    # This function takes an input audio file, converts it to text, then back to speech
    process_audio_with_mms(input_audio_path, output_audio_path)

# Example usage
if __name__ == "__main__":
    # Input audio file path (change this to the location of your audio file)
    input_audio_path = "./test-input.wav"
    # Output audio file path where the converted audio will be saved
    output_audio_path = "./test-output.wav"
    
    # Run the main function to perform the conversion (speech to text -> text to speech)
    main(input_audio_path, output_audio_path)
    
    # Print a message indicating completion
    print("Speech to text to speech conversion completed.")
