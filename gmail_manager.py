#!/usr/bin/env python3
"""
Gmail Manager for PhD Outreach Automation
Handles Gmail API integration for sending personalized emails to professors
"""

import os
import base64
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pickle
from pathlib import Path

# Gmail API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Email composition imports
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

logger = logging.getLogger(__name__)

class GmailManager:
    """Manages Gmail API operations for sending PhD outreach emails."""
    
    def __init__(self, credentials_file: str = "credentials.json"):
        self.credentials_file = credentials_file
        self.token_file = "gmail_token.pickle"
        self.scopes = [
            'https://www.googleapis.com/auth/gmail.send',
            'https://www.googleapis.com/auth/gmail.readonly'
        ]
        self.service = None
        self.user_email = None
        self._initialize_service()
    
    def _initialize_service(self) -> bool:
        """Initialize Gmail service with authentication and SSL error handling."""
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(self.token_file):
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)
            
            # If there are no valid credentials, get them
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        logger.warning(f"Token refresh failed: {e}, getting new token")
                        creds = None
                
                if not creds:
                    if not os.path.exists(self.credentials_file):
                        logger.error(f"Credentials file not found: {self.credentials_file}")
                        return False
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.scopes)
                    creds = flow.run_local_server(port=0)
                
                # Save credentials for next run
                with open(self.token_file, 'wb') as token:
                    pickle.dump(creds, token)
            
            # Build the service with credentials
            self.service = build('gmail', 'v1', credentials=creds, cache_discovery=False)
            
            # Get user email
            profile = self.service.users().getProfile(userId='me').execute()
            self.user_email = profile.get('emailAddress', '')
            
            logger.info(f"Gmail service initialized for: {self.user_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gmail service: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Gmail API connection."""
        try:
            if not self.service:
                return {"success": False, "error": "Gmail service not initialized"}
            
            profile = self.service.users().getProfile(userId='me').execute()
            return {
                "success": True,
                "email": profile.get('emailAddress'),
                "total_messages": profile.get('messagesTotal', 0),
                "threads_total": profile.get('threadsTotal', 0)
            }
        except Exception as e:
            logger.error(f"Gmail connection test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def create_email_message(self, to_email: str, subject: str, body: str, 
                           from_name: str, cv_path: str = "") -> MIMEMultipart:
        """Create email message with optional CV attachment."""
        
        # Create message
        message = MIMEMultipart()
        message['to'] = to_email
        message['subject'] = subject
        
        # Ensure user_email is a string
        user_email = self.user_email if self.user_email else "unknown@example.com"
        message['from'] = f"{from_name} <{user_email}>" if from_name else user_email
        
        # Add body
        body_part = MIMEText(body, 'plain', 'utf-8')
        message.attach(body_part)
        
        # Add CV attachment if provided and file exists
        if cv_path and os.path.exists(cv_path):
            try:
                with open(cv_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(cv_path)}'
                )
                message.attach(part)
                logger.info(f"CV attached: {cv_path}")
                
            except Exception as e:
                logger.warning(f"Failed to attach CV: {e}")
        
        return message
    
    def send_email(self, to_email: str, subject: str, body: str, 
                   from_name: str, cv_path: str = "") -> Dict[str, Any]:
        """Send a single email to a professor with retry logic for SSL issues."""
        
        if not self.service:
            return {"success": False, "error": "Gmail service not initialized"}
        
        # Validate CV path - convert None to empty string
        if cv_path is None:
            cv_path = ""
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Create email message
                message = self.create_email_message(to_email, subject, body, from_name, cv_path)
                
                # Convert to Gmail format
                raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
                gmail_message = {'raw': raw_message}
                
                # Send email
                result = self.service.users().messages().send(
                    userId='me', 
                    body=gmail_message
                ).execute()
                
                message_id = result.get('id')
                logger.info(f"Email sent successfully to {to_email}, Message ID: {message_id}")
                
                return {
                    "success": True,
                    "message_id": message_id,
                    "to_email": to_email,
                    "sent_at": datetime.now().isoformat()
                }
                
            except HttpError as e:
                error_msg = f"Gmail API error: {e}"
                logger.error(error_msg)
                
                # Check if it's a quota/rate limit error
                if hasattr(e, 'resp') and e.resp.status in [429, 403]:
                    if attempt < max_retries - 1:
                        logger.info(f"Rate limit hit, waiting {retry_delay * (attempt + 1)} seconds before retry {attempt + 1}")
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                
                return {"success": False, "error": error_msg}
            
            except Exception as e:
                error_msg = f"Email send error: {e}"
                logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                
                # Check for SSL/connection errors that can be retried
                if any(ssl_error in str(e).lower() for ssl_error in [
                    'eof occurred in violation of protocol', 
                    'connection reset', 
                    'ssl', 
                    'timeout',
                    'connection aborted'
                ]):
                    if attempt < max_retries - 1:
                        logger.info(f"SSL/Connection error detected, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        
                        # Reinitialize service on SSL errors
                        try:
                            self._initialize_service()
                        except Exception as init_error:
                            logger.warning(f"Service reinitialization failed: {init_error}")
                        
                        continue
                
                # If it's not a retryable error, or we've exhausted retries
                return {"success": False, "error": error_msg}
        
        return {"success": False, "error": f"Failed after {max_retries} attempts"}
    
    def send_bulk_emails(self, email_data_list: List[Dict[str, str]], 
                        from_name: str, cv_path: str = "", 
                        delay_seconds: int = 5) -> List[Dict[str, Any]]:
        """Send emails to multiple professors with rate limiting."""
        
        results = []
        
        for i, email_data in enumerate(email_data_list):
            logger.info(f"Sending email {i+1}/{len(email_data_list)} to {email_data['to_email']}")
            
            result = self.send_email(
                to_email=email_data['to_email'],
                subject=email_data['subject'],
                body=email_data['body'],
                from_name=from_name,
                cv_path=cv_path
            )
            
            # Add additional info
            result.update({
                "professor_id": email_data.get('professor_id'),
                "professor_name": email_data.get('professor_name'),
                "university": email_data.get('university')
            })
            
            results.append(result)
            
            # Rate limiting - wait between emails
            if i < len(email_data_list) - 1 and delay_seconds > 0:
                import time
                time.sleep(delay_seconds)
        
        return results
    
    def get_sent_email_info(self, message_id: str) -> Dict[str, Any]:
        """Get information about a sent email."""
        
        if not self.service:
            return {"success": False, "error": "Gmail service not initialized"}
        
        try:
            message = self.service.users().messages().get(
                userId='me', 
                id=message_id,
                format='metadata'
            ).execute()
            
            headers = {h['name']: h['value'] for h in message['payload']['headers']}
            
            return {
                "success": True,
                "message_id": message_id,
                "to": headers.get('To'),
                "subject": headers.get('Subject'),
                "date": headers.get('Date'),
                "thread_id": message.get('threadId')
            }
            
        except Exception as e:
            logger.error(f"Error getting email info: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_email_address(self, email: str) -> bool:
        """Validate email address format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def create_email_preview(self, to_email: str, subject: str, body: str, 
                           from_name: str) -> str:
        """Create a preview of the email for review."""
        
        preview = f"""
FROM: {from_name} <{self.user_email}>
TO: {to_email}
SUBJECT: {subject}

{body}

---
Attachments: CV (if uploaded)
        """
        return preview.strip()


class EmailTemplateManager:
    """Manages email templates for different scenarios."""
    
    @staticmethod
    def get_professional_template() -> str:
        """Get professional PhD application email template."""
        return """Dear Professor {professor_name},

I hope this email finds you well. I am writing to express my strong interest in pursuing a PhD position in your research group at {university}.

{research_alignment_paragraph}

My background includes {user_background}, and I believe my experience in {relevant_skills} aligns well with your current research directions. I am particularly excited about the opportunity to contribute to {specific_research_area}.

I have attached my CV for your review and would be grateful for the opportunity to discuss potential research opportunities in your lab. I would be happy to provide additional materials or answer any questions you might have.

Thank you for your time and consideration. I look forward to hearing from you.

Best regards,
{user_name}

---
Attachment: CV
"""
    
    @staticmethod
    def get_concise_template() -> str:
        """Get concise PhD application email template."""
        return """Dear Professor {professor_name},

I am {user_name}, interested in pursuing a PhD in your research group at {university}.

{research_alignment_paragraph}

I believe my background in {user_background} and skills in {relevant_skills} would contribute to your research in {specific_research_area}.

Please find my CV attached. I would appreciate the opportunity to discuss potential research opportunities.

Best regards,
{user_name}
"""
    
    @staticmethod
    def get_research_focused_template() -> str:
        """Get research-focused PhD application email template."""
        return """Dear Professor {professor_name},

I am writing to inquire about PhD opportunities in your research group at {university}. Your work on {specific_research_area} strongly aligns with my research interests and career goals.

{research_alignment_paragraph}

My research experience in {user_background} has prepared me to contribute meaningfully to your ongoing projects. I am particularly interested in {collaboration_potential}.

I have attached my CV and would welcome the opportunity to discuss how my background could contribute to your research program.

Thank you for your consideration.

Sincerely,
{user_name}
"""


def format_email_body(template: str, professor_data: dict, user_data: dict) -> str:
    """Format email body with professor and user data."""
    try:
        return template.format(
            professor_name=professor_data.get('name', 'Professor'),
            university=professor_data.get('university', ''),
            research_alignment_paragraph=professor_data.get('collaboration_potential', ''),
            user_background=user_data.get('background', ''),
            relevant_skills=user_data.get('skills', ''),
            specific_research_area=professor_data.get('research_interests', ''),
            collaboration_potential=professor_data.get('collaboration_potential', ''),
            user_name=user_data.get('name', '')
        )
    except KeyError as e:
        logger.warning(f"Missing template variable: {e}")
        return template


def create_bulk_email_data(professors_list: List[dict], user_name: str) -> List[Dict[str, str]]:
    """Create bulk email data from professors list."""
    
    bulk_data = []
    
    for prof in professors_list:
        if prof.get('email') and prof.get('draft_email_subject') and prof.get('draft_email_body'):
            bulk_data.append({
                'professor_id': prof.get('id'),
                'professor_name': prof.get('name'),
                'university': prof.get('university'),
                'to_email': prof['email'],
                'subject': prof['draft_email_subject'],
                'body': prof['draft_email_body']
            })
    
    return bulk_data


def test_gmail_setup(credentials_file: str = "credentials.json") -> Dict[str, Any]:
    """Test Gmail setup and return status."""
    
    try:
        gmail_manager = GmailManager(credentials_file)
        
        if gmail_manager.service:
            connection_test = gmail_manager.test_connection()
            
            if connection_test["success"]:
                return {
                    "success": True,
                    "message": f"Gmail connected successfully for {connection_test['email']}",
                    "email": connection_test['email'],
                    "total_messages": connection_test.get('total_messages', 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"Connection test failed: {connection_test['error']}"
                }
        else:
            return {
                "success": False,
                "error": "Failed to initialize Gmail service"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Gmail setup test failed: {e}"
        }


if __name__ == "__main__":
    # Test Gmail setup when run directly
    test_result = test_gmail_setup()
    
    if test_result["success"]:
        print(f"✅ Gmail setup successful!")
        print(f"📧 Connected email: {test_result['email']}")
        print(f"📊 Total messages: {test_result['total_messages']}")
    else:
        print(f"❌ Gmail setup failed: {test_result['error']}")
        print("\n📋 Troubleshooting steps:")
        print("1. Ensure credentials.json exists in the current directory")
        print("2. Make sure Gmail API is enabled in Google Cloud Console")
        print("3. Check that OAuth 2.0 credentials are correctly configured")
        print("4. Run the script and complete the authentication flow")