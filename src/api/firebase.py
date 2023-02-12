import os
import json
from dotenv import load_dotenv

from firebase_admin import credentials, firestore, initialize_app

load_dotenv()

# Initialize Firestore DB
service_account = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
cred = credentials.Certificate(service_account)
firebase_url = os.environ.get('FIREBASE_URL')
default_app = initialize_app(cred, options={"databaseURL": firebase_url})
db = firestore.client()
articles_ref = db.collection('articles')
