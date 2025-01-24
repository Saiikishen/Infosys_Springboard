import os
import uuid
import logging
from flask_cors import CORS
from flask import Flask, request, send_file, jsonify,render_template
from pydub import AudioSegment
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import ffmpeg

app = Flask(__name__)
CORS(app)
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'PROCESSED_FOLDER': 'processed',
    'TEMP_FOLDER': 'temp',
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024,
    'ALLOWED_EXTENSIONS': {'mp4', 'mov', 'avi', 'mkv'}
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video(file_path, target_lang):
    temp_audio_path = None
    translated_audio_path = None
    
    try:
        # 1. Audio extraction
        logger.info("Extracting audio...")
        try:
            audio = AudioSegment.from_file(file_path)
            if audio.duration_seconds == 0:
                logger.error("No audio track found")
                return None
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            return None

        temp_audio_path = os.path.join(app.config['TEMP_FOLDER'], f"{uuid.uuid4()}.wav")
        audio.export(temp_audio_path, format="wav")

        # 2. Speech to text
        logger.info("Converting speech to text...")
        r = sr.Recognizer()
        try:
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language="en")
                logger.info(f"Recognized text: {text[:200]}...")
        except sr.UnknownValueError:
            logger.error("Google couldn't understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"API Error: {e}")
            return None

        # 3. Translation
        logger.info("Translating text...")
        try:
            translator = Translator()
            translated = translator.translate(text, dest=target_lang)
            translated_text = translated.text
            logger.info(f"Translated text: {translated_text[:200]}...")
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return None

        # 4. Text to speech
        logger.info("Generating translated audio...")
        try:
            tts = gTTS(translated_text, lang=target_lang)
            translated_audio_path = os.path.join(app.config['TEMP_FOLDER'], f"{uuid.uuid4()}.mp3")
            tts.save(translated_audio_path)
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")
            return None

        # 5. Video merging
        logger.info("Merging audio with video...")
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{uuid.uuid4()}.mp4")

        try:
            input_video = ffmpeg.input(file_path)
            input_audio = ffmpeg.input(translated_audio_path)

            output = ffmpeg.output(
                input_video['v:0'],
                input_audio['a:0'],
                output_path,
                vcodec='libx264',
                acodec='aac',
                pix_fmt='yuv420p',
                movflags='+faststart',
                crf=23,
                preset='fast',
                strict='experimental'
            ).overwrite_output()

            ffmpeg.merge_outputs(output).run(
                capture_stdout=True,
                capture_stderr=True
            )
            logger.info(f"Video created at: {output_path}")
            return output_path

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown FFmpeg error"
            logger.error(f"FFmpeg merge failed: {error_msg}")
            return None
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        return None
    finally:
        # Cleanup with existence checks
        for path in [temp_audio_path, translated_audio_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up: {path}")
                except Exception as e:
                    logger.warning(f"Cleanup failed for {path}: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    target_lang = request.form.get('lang', 'en').lower()

    if not file or file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result_path = process_video(file_path, target_lang)
        
        if result_path and os.path.exists(result_path):
            return jsonify({
                "download_url": f"/download/{os.path.basename(result_path)}"
            })
        return jsonify({"error": "Processing failed"}), 500

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({"error": "Server error"}), 500
    

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        logger.info(f"Attempting to send file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404

        return send_file(
            file_path,
            as_attachment=True,
            mimetype='video/mp4'
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}", exc_info=True)
        return jsonify({"error": "Download error"}), 500

if __name__ == '__main__':
    for folder in ['UPLOAD_FOLDER', 'PROCESSED_FOLDER', 'TEMP_FOLDER']:
        os.makedirs(app.config[folder], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)