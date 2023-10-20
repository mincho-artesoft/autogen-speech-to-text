
from flask import Flask, render_template, request, render_template_string
import soundfile as sf
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import librosa
import json
import openai


openai.api_key = 'sk-'

app = Flask(__name__, template_folder="templates")

model = Speech2TextForConditionalGeneration.from_pretrained(
    "facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained(
    "facebook/s2t-small-librispeech-asr")


def convert_sampling_rate(audio_data, current_rate):
    return librosa.resample(audio_data, orig_sr=current_rate, target_sr=16000), 16000


@app.route('/', methods=['GET', 'POST'])
def index():
    transcription = ""
    extra_string = ""
    combined_string = ""
    response_text = ""
    if request.method == 'POST':
        audio_file = request.files['file']
        extra_string = request.form['extra_string']
        audio_data, samplerate = sf.read(audio_file)
        audio_data_resampled, samplerate = convert_sampling_rate(
            audio_data, samplerate)
        inputs = processor(
            audio_data_resampled, sampling_rate=samplerate, return_tensors="pt", padding=True)
        generated_ids = model.generate(**inputs)
        transcription = processor.decode(generated_ids[0])
        transcription = transcription.replace('</s>', '')
        print(transcription)
        print(extra_string)
        combined_string = f"Use whatever makes more sense to you. 1: \n{transcription}\n. 2: \n{extra_string}\n."
        print(combined_string)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": combined_string}]
        )
        print(response)
        response_text = response['choices'][0]['message']['content']
    return render_template('index.html', transcription=response_text)


if __name__ == '__main__':
    app.run(debug=True)

"""   
      response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": combined_string}]
        )
        print(response)
        response_text = response['choices'][0]['message']['content']
          """