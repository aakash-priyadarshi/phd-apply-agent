from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os

# Correct scopes for Gmail sending
SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.readonly'
]


def test_gmail_auth():
    creds = None

    # Load existing token
    if os.path.exists('gmail_token.pickle'):
        with open('gmail_token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If there are no valid credentials, get them
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials
        with open('gmail_token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Test Gmail service
    service = build('gmail', 'v1', credentials=creds)
    profile = service.users().getProfile(userId='me').execute()

    print(f"✅ Gmail authenticated for: {profile.get('emailAddress')}")
    print(f"✅ Total messages: {profile.get('messagesTotal', 'Unknown')}")
    return True


if __name__ == "__main__":
    test_gmail_auth()
