import os
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template
from flask_cors import CORS
from whisper import audio_log
from pydub import AudioSegment
import time
from datetime import datetime

import whisper
import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Google Docs API Call

# Replace with your desired API scope
SCOPES = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/documents.readonly']
CREDENTIALS_FILE = 'token.pickle'


# The ID of a sample document.
DOCUMENT_ID = '11NCLCwpi2vlpksIQJIQ3222gRlxfa_BtryAXnhO6xEM'
BASE_URL = "https://docs.google.com/document/d/"
url = f"{BASE_URL}{DOCUMENT_ID}"


print(url)
def create_google_doc(text):
    """Shows basic usage of the Docs API.
    Prints the title of a sample document.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        print("test")
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('docs', 'v1', credentials=creds)

        # Retrieve the documents contents from the Docs service.
        document = service.documents().get(documentId=DOCUMENT_ID).execute()
        # doc_dict = json.loads(document)
        # Define the requests to insert the text
        requests = [
            {
                'insertText': {
                    'location': {
                        'index': 1

                    },
                    'text': text
                }
            }
        ]

        # Execute the requests
        result = service.documents().batchUpdate(documentId=DOCUMENT_ID, body={'requests': requests}).execute()

        print('The title of the document is: {}'.format(document.get('title')))
        print(result)
    except HttpError as err:
        print(err)


# Initialize Flask app
app = Flask(__name__, static_folder='templates')
# Add upload folder
data_dir = os.path.join(os.path.dirname(__file__), 'templates')
CORS(app)
convertion_name = "Voice of " + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

# Route for serving the index page
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Route for handling file upload and processing
@app.route('/upload', methods=['POST' , "GET"])
def upload():
    audio_file = AudioSegment.from_file(request.files['audio'])
    # # # ## Convert file audio to mp3 format
    audio_file.export(os.path.join(data_dir, 'converted.mp3'), format="mp3")
    audio =open(os.path.join(data_dir, 'converted.mp3'), 'rb')
    audio_data = audio.read()
    result = audio_log(audio_data)
    result_text = f"{convertion_name} \n \n \n {result}. \n \n \n"
    create_google_doc(result_text)
    return jsonify({"result": url})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

