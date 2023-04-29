import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from whisper import audio_log
import whisper
import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Google Docs API Call

# Replace with your desired API scope
SCOPES = ['https://www.googleapis.com/auth/documents']
CREDENTIALS_FILE = 'token.pickle'

def get_credentials():
    creds = None

    # Load credentials from file if it exists
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'rb') as token:
            creds = pickle.load(token)

    # Refresh or create new credentials if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config({
                'installed': {
                    'client_id': os.environ['GOOGLE_CLIENT_ID'],
                    'client_secret': os.environ['GOOGLE_CLIENT_SECRET'],
                    'redirect_uris': "http://127.0.0.1:5000", # os.environ['GOOGLE_REDIRECT_URI'],
                    'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                    'token_uri': 'https://accounts.google.com/o/oauth2/token',
                }
            }, SCOPES)

            creds = flow.run_local_server(port=0)
        
        # Save credentials to file
        with open(CREDENTIALS_FILE, 'wb') as token:
            pickle.dump(creds, token)

    return creds

# Function to create a Google Doc with the audio log results
def create_google_doc(result):
    try:
        credentials = get_credentials()  # Use the get_credentials() function
        service = build('docs', 'v1', credentials=credentials)

        body = {
            'title': 'Audio Log Results'
        }
        doc = service.documents().create(body=body).execute()
        print('Created document with title: {0}'.format(doc.get('title')))

        document_id = doc['documentId']
        requests = [
            {
                'insertText': {
                    'location': {
                        'index': 1,
                    },
                    'text': result
                }
            }
        ]
        result = service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests}).execute()

    except HttpError as error:
        print('An error occurred: %s' % error)
        result = None


# Initialize Flask app
app = Flask(__name__, static_folder='templates')
data_dir = os.path.join(os.path.dirname(__file__), 'uploads')

CORS(app)

# Route for serving the index page
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Route for handling file upload and processing
@app.route('/upload', methods=['POST'])
def upload():
    audio_file = request.files['audio'].read()
    audio_file = AudioSegment.from_file(request.files['audio'])
    ## Convert file audio to mp3 format
    audio_file.export(os.path.join(data_dir, 'converted.mp3'), format="mp3")
    audio = whisper.load_audio(os.path.join(data_dir, 'converted.mp3'))
    result = audio_log(audio)
    # create_google_doc(result)
    return jsonify({"result": result})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

