import requests
import time
import os 
from dotenv import load_dotenv

load_dotenv()

transcript_endpoint = 'https://api.assemblyai.com/v2/transcript'
headers = {
    "authorization": os.getenv('ASSEMBLYAI_API_KEY')
}

CHUNK_SIZE = 5_242_880  # 5MB


def upload_to_assemblyai(file):
    upload_endpoint = 'https://api.assemblyai.com/v2/upload'

    def read_file(file_obj):
        while True:
            data = file_obj.read(CHUNK_SIZE)
            if not data:
                break
            yield data

    response = requests.post(upload_endpoint, headers=headers, data=read_file(file))
    response.raise_for_status()
    return response.json()['upload_url']


def transcribe_and_poll(audio_url):
    transcript_request = {'audio_url': audio_url}
    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
    transcript_response.raise_for_status()
    transcript_id = transcript_response.json()['id']

    while True:
        polling_response = requests.get(f'{transcript_endpoint}/{transcript_id}', headers=headers)
        polling_response.raise_for_status()
        result = polling_response.json()

        if result['status'] == 'completed':
            return result['text']
        elif result['status'] == 'error':
            return f"Error: {result['error']}"
        time.sleep(15)