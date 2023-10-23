from flask import Flask, render_template, request, render_template_string
import soundfile as sf
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import librosa
import json
import openai
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

openai.api_key = 'sk-'

app = Flask(__name__, template_folder="templates")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    chunk_length_s=30,
    device=device,
)

pipe.model.config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="english", task="translate")

def convert_sampling_rate(audio_data, current_rate):
    return librosa.resample(audio_data, orig_sr=current_rate, target_sr=16000), 16000


@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = ""
    prediction = ""
    if request.method == 'POST':
        audio_file = request.files['file']
        audio_data, samplerate = sf.read(audio_file)
        audio_data_resampled, samplerate = convert_sampling_rate(
            audio_data, samplerate)

        prediction = pipe(audio_data_resampled.copy(), batch_size=8)["text"]
        print(prediction)

    return render_template('index.html', transcription=prediction)


if __name__ == '__main__':
    app.run(debug=True)

    """         response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prediction}]
        )
        print(response)
        response_text = response['choices'][0]['message']['content'] """
