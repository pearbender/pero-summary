from flask import Flask, render_template_string
from twitchrealtimehandler import TwitchAudioGrabber
import numpy as np
from io import BytesIO
from pydub import AudioSegment
from pydub.exceptions import CouldntEncodeError
from pydub.silence import detect_nonsilent
from openai import OpenAI
import threading
import logging
import queue

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the format of the log messages
    datefmt='%Y-%m-%d %H:%M:%S',  # Define the date format
)

logging.getLogger('faster_whisper').setLevel(logging.ERROR)

app = Flask(__name__)
current_summary = ""
temp_file_path = 'data/temp.wav'
q = queue.Queue()

def get_segments(data):
    if 'segments' in data:
        for segment in data['segments']:
            yield segment
    else:
        for segment in data[0]:
            yield {'start': segment.start, 'end': segment.end, 'text': segment.text}

def capture_audio():
    audio_grabber = TwitchAudioGrabber(twitch_url='https://www.twitch.tv/perokichi_neet', dtype=np.int16, segment_length=10, channels=1, rate=16000)
    audio = AudioSegment.silent(duration=0)
    while True:
        audio_segment = audio_grabber.grab_raw()
        if not audio_segment:
            continue
        raw = BytesIO(audio_segment)
        try:
            raw_wav = AudioSegment.from_raw(raw, sample_width=2, frame_rate=16000, channels=1)
        except CouldntEncodeError:
            logging.error("Could not encode new audio.")
            continue
        audio += raw_wav
        if audio.duration_seconds < 120:
            continue
        logging.info("Passing audio...")
        q.put(audio[-120000:])
        audio = audio[-60000:]

def process_audio():
    global current_summary
    client = OpenAI()
    while True:
        last_item = None
        while not q.empty():
                last_item = q.get()
                q.task_done()
        if last_item is None:
             continue
        logging.info("Trascribing audio...")
        last_item.export(temp_file_path)
        audio_file= open(temp_file_path, "rb")
        translation = client.audio.translations.create(
                model="whisper-1", 
                file=audio_file,
                prompt="PearBender welcome, fuck english... ええと こんばんは、Alex welcome, いやねぇ 今日はねぇ あ ところでさ ごめん あの 今日ねぇ 今日ねぇ みんな 今日ねぇ, cha- cha- what is it? cha- chazay? あとなんか変な味がする… 口の中, Pero welcome, Mathew welcome, Ender welcome. I'm playing wa- i don't get it wa- warhammer",
        )
        logging.info("Summarizing audio...")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
	    max_tokens=20,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please summarize what the streamer Pero is doing in once sentence, like \"Pero is...\", based on the following transcription. Use commas instead of bullet points. Transcription: " + translation.text}
            ]
        )
        current_summary = completion.choices[0].message.content
        logging.info("New summary: " + current_summary[:100] + '...')

@app.route('/summary', methods=['GET'])
def get_summary():
    return current_summary[:100] + 'This is a test...'  # Return plain text response

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
            @font-face {
                font-family: 'MisakiMincho';
                src: url('/static/misaki_mincho.ttf') format('truetype');
                font-weight: normal;
                font-style: normal;
            }
            #summary {
                font-family: 'MisakiMincho';
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
    threading.Thread(target=capture_audio).start()
    threading.Thread(target=process_audio).start()
    app.run(host='0.0.0.0', port=80)
