from flask import Flask, render_template_string
from twitchrealtimehandler import TwitchAudioGrabber
import numpy as np
from io import BytesIO
from pydub import AudioSegment
from pydub.exceptions import CouldntEncodeError
from pydub.silence import detect_nonsilent
from faster_whisper import WhisperModel
import torch
from openai import OpenAI
import threading
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the format of the log messages
    datefmt='%Y-%m-%d %H:%M:%S',  # Define the date format
)

logging.getLogger('faster_whisper').setLevel(logging.ERROR)

app = Flask(__name__)
current_summary = ""

def get_segments(data):
    if 'segments' in data:
        for segment in data['segments']:
            yield segment
    else:
        for segment in data[0]:
            yield {'start': segment.start, 'end': segment.end, 'text': segment.text}

def process_audio():
    global current_summary
    audio_grabber = TwitchAudioGrabber(twitch_url='https://www.twitch.tv/perokichi_neet', dtype=np.int16, segment_length=5, channels=1, rate=16000)
    audio = AudioSegment.silent(duration=0)
    model = WhisperModel('medium', device="cuda" if torch.cuda.is_available() else "cpu")
    client = OpenAI()
    while True:
        audio_segment = audio_grabber.grab_raw()
        if not audio_segment:
            continue
        raw = BytesIO(audio_segment)
        try:
            raw_wav = AudioSegment.from_raw(raw, sample_width=2, frame_rate=16000, channels=1)
        except CouldntEncodeError:
            continue
        audio += raw_wav
        if audio.duration_seconds < 60:
            continue
        logging.info("Processing new audio...")
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=800, silence_thresh=-45, seek_step=100)
        text = ""
        for start, end in nonsilent_ranges:
            temp_file_path = 'data/temp.wav'
            segment = audio[start:end]
            segment.export(temp_file_path)
            data = model.transcribe(temp_file_path,
                                    language='ja',
                                    initial_prompt="PearBender welcome, fuck english... ええと こんばんは、Alex welcome, いやねぇ 今日はねぇ あ ところでさ ごめん あの 今日ねぇ 今日ねぇ みんな 今日ねぇ, cha- cha- what is it? cha- chazay? あとなんか変な味がする… 口の中, Pero welcome, Mathew welcome, Ender welcome. I'm playing wa- i don't get it wa- warhammer",
                                    beam_size=5,
                                    best_of=1,
                                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                    suppress_tokens=[],
                                    vad_filter=True,
                                    vad_parameters={'min_silence_duration_ms': 800}
                                    )
            segments = list(get_segments(data))
            for segment in segments:
                text += segment['text']
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please summarize \"What the streamer Pero is doing\" like \"Eating breakfast,\" \"Looking for food,\" etc. without including subject based on the following transcription: " + text}
            ]
        )
        current_summary = completion.choices[0].message.content
        logging.info("New summary: " + current_summary)
        audio = audio[-30000:]

@app.route('/summary', methods=['GET'])
def get_summary():
    return 'Pero is: ' + current_summary  # Return plain text response

@app.route('/')
def index():
    # Simplified HTML that only displays the summary text in Orbitron font
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Press Start 2P', sans-serif;
            }
        </style>
        <script>
            function refreshSummary() {
                fetch('/summary')
                    .then(response => response.text())
                    .then(data => {
                        document.getElementById('summary').innerText = data;
                    });
            }
            setInterval(refreshSummary, 2000); // Refresh every 2 seconds
            window.onload = refreshSummary;
        </script>
    </head>
    <body>
        <div id="summary"></div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    threading.Thread(target=process_audio).start()
    app.run(host='0.0.0.0', port=80)
