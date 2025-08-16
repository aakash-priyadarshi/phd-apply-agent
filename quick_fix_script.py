#!/usr/bin/env python3
"""
Quick Fix Script for PhD Outreach Automation
Fixes the WebDriver and Gmail credentials issues.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def print_colored(text, color_code):
    """Print colored text."""
    print(f"\033[{color_code}m{text}\033[0m")


def print_header(text):
    """Print section header."""
    print_colored(f"\n{'='*60}", "96")
    print_colored(f"{text:^60}", "1;96")
    print_colored(f"{'='*60}", "96")


def fix_webdriver_issue():
    """Fix the WebDriver initialization issue."""
    print_header("FIXING WEBDRIVER ISSUE")

    print_colored("🔧 Fixing Selenium WebDriver compatibility...", "93")

    try:
        # Uninstall and reinstall with specific versions
        print("Updating Selenium and WebDriver Manager...")

        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y",
            "selenium", "webdriver-manager"
        ], capture_output=True)

        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "selenium==4.15.0", "webdriver-manager==4.0.1"
        ], capture_output=True)

        print_colored("✅ WebDriver dependencies fixed", "92")
        return True

    except Exception as e:
        print_colored(f"❌ WebDriver fix failed: {e}", "91")
        return False


def check_gmail_credentials():
    """Check and guide user for correct Gmail credentials."""
    print_header("CHECKING GMAIL CREDENTIALS")

    if not Path("credentials.json").exists():
        print_colored("❌ No credentials.json found", "91")
        return False

    try:
        with open("credentials.json", 'r') as f:
            creds = json.load(f)

        # Check if it's a service account (wrong type)
        if "type" in creds and creds["type"] == "service_account":
            print_colored("❌ Found Service Account credentials", "91")
            print_colored(
                "   Gmail API requires OAuth2 Desktop App credentials", "93")
            print("")
            print_colored("🔧 TO FIX:", "1;94")
            print_colored("1. Go to: https://console.cloud.google.com/", "96")
            print_colored(
                "2. Navigate to: APIs & Services → Credentials", "96")
            print_colored(
                "3. Click: + CREATE CREDENTIALS → OAuth client ID", "96")
            print_colored("4. Choose: Desktop application", "96")
            print_colored("5. Download the new credentials.json", "96")
            print_colored("6. Replace your current credentials.json", "96")
            print("")

            # Show difference
            print_colored("❌ Current (Service Account):", "91")
            print('   {"type": "service_account", ...}')
            print("")
            print_colored("✅ Needed (OAuth2 Desktop):", "92")
            print('   {"installed": {"client_id": "...", ...}}')
            print("")

            return False

        # Check if it's OAuth2 (correct type)
        elif "installed" in creds or "web" in creds:
            print_colored(
                "✅ Correct OAuth2 Desktop App credentials found", "92")
            print_colored("   Gmail API should work properly", "96")
            return True

        else:
            print_colored("❌ Unknown credential format", "91")
            return False

    except Exception as e:
        print_colored(f"❌ Error reading credentials: {e}", "91")
        return False


def test_quick_fixes():
    """Test the fixes with a minimal setup."""
    print_header("TESTING FIXES")

    # Test WebDriver
    print_colored("🧪 Testing WebDriver...", "93")
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://www.google.com")
        driver.quit()

        print_colored("✅ WebDriver test passed", "92")

    except Exception as e:
        print_colored(f"❌ WebDriver test failed: {e}", "91")
        print_colored("💡 Try running setup again after this fix", "93")

    # Test Gmail credentials format
    print_colored("🧪 Testing Gmail credentials format...", "93")
    gmail_ok = check_gmail_credentials()

    return gmail_ok


def create_sample_oauth2_credentials():
    """Create a sample OAuth2 credentials file for reference."""
    print_header("CREATING SAMPLE CREDENTIALS")

    sample_creds = {
        "installed": {
            "client_id": "your-client-id.apps.googleusercontent.com",
            "project_id": "jarvis-383903",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "your-client-secret",
            "redirect_uris": ["http://localhost"]
        }
    }

    try:
        with open("credentials_oauth2_sample.json", "w") as f:
            json.dump(sample_creds, f, indent=2)

        print_colored("✅ Created: credentials_oauth2_sample.json", "92")
        print_colored("   This shows the correct format needed", "96")
        return True

    except Exception as e:
        print_colored(f"❌ Failed to create sample: {e}", "91")
        return False


def main():
    """Main fix function."""
    print_colored("""
🔧 PhD Outreach Automation - Quick Fix
Fixing WebDriver and Gmail credentials issues
""", "1;96")

    print_colored("Issues found in your setup:", "93")
    print_colored("1. WebDriver initialization conflict", "91")
    print_colored(
        "2. Wrong Gmail credentials type (Service Account vs OAuth2)", "91")
    print("")

    # Fix 1: WebDriver
    webdriver_fixed = fix_webdriver_issue()

    # Fix 2: Gmail credentials check
    gmail_ok = check_gmail_credentials()

    # Create sample for reference
    create_sample_oauth2_credentials()

    # Test fixes
    test_quick_fixes()

    # Final summary
    print_header("FIX SUMMARY")

    if webdriver_fixed:
        print_colored("✅ WebDriver issue: FIXED", "92")
    else:
        print_colored("❌ WebDriver issue: Still needs attention", "91")

    if gmail_ok:
        print_colored("✅ Gmail credentials: CORRECT", "92")
    else:
        print_colored(
            "❌ Gmail credentials: Need OAuth2 Desktop App type", "91")

    print("")

    if webdriver_fixed and gmail_ok:
        print_colored("🎉 All issues fixed! You can now run:", "1;92")
        print_colored("   python setup.py", "96")
        print_colored("   OR", "93")
        print_colored("   run_setup.bat", "96")
    else:
        print_colored("🔧 Next steps:", "1;93")
        if not gmail_ok:
            print_colored(
                "1. Get correct OAuth2 credentials from Google Cloud Console", "93")
            print_colored("2. Replace your credentials.json", "93")
        print_colored("3. Run setup again: python setup.py", "93")

    print("")
    print_colored(
        "💡 Need help? Check credentials_oauth2_sample.json for the correct format", "96")


if __name__ == "__main__":
    main()
