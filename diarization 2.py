import os
import re
import subprocess
import requests
import json
import time
from pydub import AudioSegment
from pyannote.audio import Pipeline

import uuid
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from langdetect import detect  # Language detection library

# Function to detect the language of the text
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Function to download audio from URL
def download_audio(url, download_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return download_path
    except Exception as e:
        print(f"Error downloading audio file: {e}")
        return None

# Function to transcribe audio file using Whisper
def transcribe_audio(file_path, model='medium', word_timestamps=True):
    command = [
        "whisper", str(file_path),
        "--model", model,
        "--word_timestamps", str(word_timestamps),
        "--output_format", "json"
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        return "", result.stderr

    return result.stdout, ""

# Function to parse the transcription text into a list of dictionaries
def parse_transcription(transcription):
    pattern = r'\[(\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{02}\.\d{3})\]  (.+)'
    matches = re.findall(pattern, transcription)
    result = []
    for match in matches:
        start_time, end_time, text = match
        result.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": text
        })
    return result

# Function to perform speaker diarization and save speaker segments
def perform_diarization(audio_file_path, output_dir, file_id):
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="YOUR_HF_AUTH_TOKEN")
    except Exception as e:
        print(f"Error loading pyannote pipeline: {e}")
        return None, None

    wav_file_path = None
    try:
        if not audio_file_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(audio_file_path)
            wav_file_path = audio_file_path.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_file_path, format='wav')
            audio_file_path = wav_file_path

        diarization = pipeline(audio_file_path)
        audio = AudioSegment.from_file(audio_file_path)
        segments = []

        for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            segment = audio[turn.start * 1000: turn.end * 1000]
            segment_path = os.path.join(output_dir, f"{file_id}_speaker_{speaker}_{i}_{turn.start:.2f}_{turn.end:.2f}.wav")
            segment.export(segment_path, format="wav")
            segments.append((segment_path, turn.start, turn.end))

        return segments, audio, wav_file_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Function to adjust transcription timestamps relative to the full audio file
def adjust_timestamps(transcriptions, segment_start, segment_end, full_audio_length):
    adjusted = []
    for entry in transcriptions:
        start_time = entry['start_time']
        end_time = entry['end_time']

        def timestamp_to_seconds(ts):
            minutes, seconds = map(float, ts.split(':'))
            return minutes * 60 + seconds

        start_seconds = timestamp_to_seconds(start_time)
        end_seconds = timestamp_to_seconds(end_time)

        adjusted_start = segment_start + start_seconds
        adjusted_end = segment_start + end_seconds

        if adjusted_end > full_audio_length:
            adjusted_end = full_audio_length

        def seconds_to_timestamp(seconds):
            minutes, seconds = divmod(seconds, 60)
            return f"{int(minutes):02}:{seconds:05.3f}"

        adjusted.append({
            "start_time": seconds_to_timestamp(adjusted_start),
            "end_time": seconds_to_timestamp(adjusted_end),
            "text": entry['text']
        })
    return adjusted

# Function to merge parsed transcriptions
def merge_transcriptions(transcriptions):
    merged = []
    for entry in transcriptions:
        # Detect the language of the current text
        language = detect_language(entry['text'])
        
        # Replace non-English text with #
        if language == 'en':
            merged.append(entry)

#        if merged and merged[-1]['end_time'] == entry['start_time']:
#            merged[-1]['text'] += " " + entry['text']
#            merged[-1]['end_time'] = entry['end_time']
#        elif merged and merged[-1]['end_time'] > entry['start_time']:
            # Handle overlap by merging texts and setting end_time to the maximum of both
#            merged[-1]['text'] += " " + entry['text']
#            merged[-1]['end_time'] = max(merged[-1]['end_time'], entry['end_time'])

        else:
            merged.append({
                "start_time": entry['start_time'],
                "end_time": entry['end_time'],
                "text": '#' * len(entry['text'])  # Use # for non-English segments
            })
    return merged

# Function to fetch mp3 url from the api
def fetch_mp3_url(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        audio_data = response.json()
        return audio_data.get('mp3url')
    except Exception as e:
        print(f"Error fetching audio URL from API: {e}")
        return None

# Function to send transcription result to API
def send_transcription(api_url, mp3_url, transcription):
    payload = {
        'mp3_url': mp3_url,
        'speech_to_text': json.dumps(transcription)
    }
    response = requests.patch(api_url, params=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to send transcription: {response.status_code}")


# Function to handle the processing of each MP3 URL
def process_mp3_url(mp3_url, set_transcription_api):
    try:
        base_directory = "temp_4"  # Replace with your desired base directory

        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        # Create a unique subdirectory for each audio file
        file_id = str(uuid.uuid4())[:8]
        unique_subdir = os.path.join(base_directory, file_id)
        os.makedirs(unique_subdir)

        audio_file_path = os.path.join(unique_subdir, "downloaded_audio.mp3")
        
        # Download the audio file to the unique subdirectory
        download_path = download_audio(mp3_url, audio_file_path)
        if not download_path:
            return

        # Perform diarization on the downloaded audio file
        segments, full_audio, wav_file_path = perform_diarization(download_path, unique_subdir, file_id)
        if segments is None:
            return

        full_audio_length = len(full_audio) / 1000.0

        combined_transcription = []

        for segment_path, segment_start, segment_end in segments:
            transcription, error = transcribe_audio(segment_path)
            if error:
                print(f"Transcription error for {segment_path}: {error}")
                continue

            parsed_transcription = parse_transcription(transcription)
            adjusted_transcription = adjust_timestamps(parsed_transcription, segment_start, segment_end, full_audio_length)

            combined_transcription.extend(adjusted_transcription)

        merged_transcription = merge_transcriptions(combined_transcription)

        print(f"Merged Transcription for {mp3_url}: {merged_transcription}")

        response = send_transcription(set_transcription_api, mp3_url, merged_transcription)
        print(f"API Response: {response}")

    except Exception as e:
        print(f"An error occurred while processing {mp3_url}: {e}")

    finally:
        # Clean up files in the unique directory if needed
        try:
            for file_path in os.listdir(unique_subdir):
                os.remove(os.path.join(unique_subdir, file_path))
            os.rmdir(unique_subdir)
        except Exception as e:
            print(f"Error cleaning up directory {unique_subdir}: {e}")



#        audio_file_path = "downloaded_audio_" + str(uuid.uuid4())[:8] + ".mp3"
#        download_path = download_audio(mp3_url, audio_file_path)
#        if not download_path:
#            return

#        segments, full_audio, wav_file_path = perform_diarization(download_path)
#        if segments is None:
#            return

#        full_audio_length = len(full_audio) / 1000.0

#        combined_transcription = []
#        created_files = []

#        try:
#            for segment_path, segment_start, segment_end in segments:
#                created_files.append(segment_path)

#                transcription, error = transcribe_audio(segment_path)
#                if error:
#                    print(f"Transcription error for {segment_path}: {error}")
#                    continue

#                parsed_transcription = parse_transcription(transcription)
#                adjusted_transcription = adjust_timestamps(parsed_transcription, segment_start, segment_end, full_audio_length)

#                combined_transcription.extend(adjusted_transcription)

#            merged_transcription = merge_transcriptions(combined_transcription)

#            print(f"Merged Transcription: {merged_transcription}")

#            response = send_transcription(set_transcription_api, mp3_url, merged_transcription)
#            print(f"API Response: {response}")

#        finally:
#            for file_path in created_files:
#                try:
#                    os.remove(file_path)
#                    print(f"Deleted file: {file_path}")
#                except Exception as e:
#                    print(f"Error deleting file {file_path}: {e}")

#            if os.path.exists(audio_file_path):
#                try:
#                    os.remove(audio_file_path)
#                    print(f"Deleted downloaded audio file: {audio_file_path}")
#                except Exception as e:
#                    print(f"Error deleting downloaded audio file {audio_file_path}: {e}")

#            if wav_file_path and os.path.exists(wav_file_path):
#                try:
#                    os.remove(wav_file_path)
#                    print(f"Deleted intermediate WAV file: {wav_file_path}")
#                except Exception as e:
#                    print(f"Error deleting intermediate WAV file {wav_file_path}: {e}")

#    except Exception as e:
#        print(f"An error occurred while processing {mp3_url}: {e}")


# function to intiate the process
def initiate_s2t(get_url_api, set_transcription_api):
    #get MP3 URL
    mp3_url = fetch_mp3_url(get_url_api)
    if mp3_url :
        process_text = [ {'start_time': '00:00', 'end_time': '00:00', 'text': 'processing'}]
        response = send_transcription(set_transcription_api, mp3_url, process_text)
        process_mp3_url(mp3_url, set_transcription_api)
    else:
        print("No new MP3 URL found or same URL as before. Sleeping for 5 seconds.")
        time.sleep(5)  # Sleep for 5 seconds


# Main function to handle downloading, diarizing, transcribing, and sending the transcription in parallel
def main():
    get_url_api = "https://tabsons-fastapi-g55rbik64q-el.a.run.app/get_first_mp3_url/"
    set_transcription_api = "https://tabsons-fastapi-g55rbik64q-el.a.run.app/set_speech_to_text"
    last_mp3_url = None
    cpu_count = multiprocessing.cpu_count()
    print("\nCPU Count:" + str(cpu_count))
    num_threads = cpu_count if cpu_count > 0 else 1  # Number of parallel threads
    num_threads = 2

    while True:
        try:
            # Running the tasks in parallel
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(initiate_s2t, get_url_api, set_transcription_api) for _ in range(num_threads)]
                for future in as_completed(futures):
                    print("\nProcessing completed.")
                    try:
                        future.result()
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"CUDA Out of Memory: {e}")
                        torch.cuda.empty_cache()  # Clear cache if OOM error occurs
                        time.sleep(5)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
