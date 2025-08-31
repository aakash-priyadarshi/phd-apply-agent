#!/usr/bin/env python3
"""
Test Gmail email sending with SSL retry logic
"""
from gmail_manager import GmailManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_email_sending():
    """Test sending a simple email to verify SSL issues are resolved."""
    try:
        # Initialize Gmail manager
        gmail_manager = GmailManager()
        
        if not gmail_manager.service:
            print("❌ Gmail service not initialized")
            return False
        
        # Test email details
        to_email = "aakashm301@gmail.com"  # Send to yourself for testing
        subject = "Test Email - PhD Outreach App SSL Fix"
        body = """Hello!

This is a test email from the PhD Outreach application to verify that the SSL connection issues have been resolved.

If you receive this email, the Gmail integration is working correctly with the new retry logic.

Best regards,
PhD Outreach App"""
        
        from_name = "PhD Outreach Test"
        
        print(f"🔄 Attempting to send test email to {to_email}...")
        
        # Send test email
        result = gmail_manager.send_email(
            to_email=to_email,
            subject=subject,
            body=body,
            from_name=from_name,
            cv_path=""
        )
        
        if result["success"]:
            print(f"✅ Test email sent successfully!")
            print(f"   Message ID: {result['message_id']}")
            print(f"   Sent at: {result['sent_at']}")
            return True
        else:
            print(f"❌ Test email failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("=== Gmail Email Sending Test ===")
    success = test_email_sending()
    
    if success:
        print("\n🎉 Gmail email sending is working correctly!")
        print("The SSL issues have been resolved with retry logic.")
    else:
        print("\n⚠️ Gmail email sending still has issues.")
        print("Check the logs above for more details.")
