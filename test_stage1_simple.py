#!/usr/bin/env python3
"""
Simple Stage 1 Test - Isolated Testing
Tests Stage 1 functionality in isolation to identify the hanging issue
"""

import os
import sys
import time
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import openai
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_colored(text, color_code):
    """Print colored text."""
    print(f"\033[{color_code}m{text}\033[0m")


def setup_simple_driver():
    """Setup a simple Chrome driver for testing."""
    print_colored("🔧 Setting up simple Chrome driver...", "93")

    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--timeout=20000")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(20)
        driver.implicitly_wait(5)

        print_colored("✅ Chrome driver setup successful", "92")
        return driver

    except Exception as e:
        print_colored(f"❌ Driver setup failed: {e}", "91")
        return None


def test_simple_openai():
    """Test OpenAI API with simple request."""
    print_colored("🧪 Testing OpenAI API...", "93")

    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print_colored("❌ No API key found", "91")
            return False

        client = openai.OpenAI(api_key=api_key)

        print_colored("Making test API request...", "96")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Just say 'API working'"}],
            max_tokens=10,
            timeout=30
        )

        result = response.choices[0].message.content
        print_colored(f"✅ API Response: {result}", "92")
        return True

    except Exception as e:
        print_colored(f"❌ API test failed: {e}", "91")
        return False


def test_simple_web_navigation(driver):
    """Test simple web navigation."""
    print_colored("🌐 Testing web navigation...", "93")

    if not driver:
        return False

    try:
        # Test 1: Simple page load
        print_colored("Testing Google homepage...", "96")
        start_time = time.time()
        driver.get("https://www.google.com")
        load_time = time.time() - start_time
        print_colored(f"✅ Google loaded in {load_time:.2f}s", "92")

        # Test 2: MIT homepage (example university)
        print_colored("Testing MIT homepage...", "96")
        start_time = time.time()
        driver.get("https://www.mit.edu")
        load_time = time.time() - start_time
        print_colored(f"✅ MIT loaded in {load_time:.2f}s", "92")

        # Test 3: Get page content
        print_colored("Testing page content extraction...", "96")
        content = driver.page_source
        content_length = len(content)
        print_colored(f"✅ Page content: {content_length} characters", "92")

        return True

    except Exception as e:
        print_colored(f"❌ Navigation test failed: {e}", "91")
        return False


def test_faculty_page_discovery(driver):
    """Test finding faculty pages (the problematic part)."""
    print_colored("🔍 Testing faculty page discovery...", "93")

    if not driver:
        return False

    try:
        university = "MIT"
        department = "Computer Science"

        # Method 1: Direct university site
        print_colored("Method 1: Direct university search...", "96")
        direct_urls = [
            "https://www.csail.mit.edu/people",
            "https://eecs.mit.edu/people/faculty",
            "https://www.mit.edu/academics/schools-departments/"
        ]

        for url in direct_urls:
            try:
                print_colored(f"Trying: {url}", "96")
                start_time = time.time()
                driver.get(url)
                load_time = time.time() - start_time
                title = driver.title[:50] + \
                    "..." if len(driver.title) > 50 else driver.title
                print_colored(f"✅ Loaded in {load_time:.2f}s: {title}", "92")

                # Quick content check
                content = driver.page_source[:1000]  # First 1000 chars
                if any(word in content.lower() for word in ["faculty", "professor", "staff"]):
                    print_colored("✅ Faculty-related content found", "92")
                else:
                    print_colored("⚠️ No faculty content detected", "93")

                time.sleep(2)  # Rate limiting

            except Exception as e:
                print_colored(f"❌ Failed to load {url}: {e}", "91")
                continue

        return True

    except Exception as e:
        print_colored(f"❌ Faculty discovery test failed: {e}", "91")
        return False


def test_minimal_stage1_process():
    """Test minimal Stage 1 process with OpenAI."""
    print_colored("🤖 Testing minimal Stage 1 AI process...", "93")

    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print_colored("❌ No API key", "91")
            return False

        client = openai.OpenAI(api_key=api_key)

        # Sample page content (shortened)
        sample_content = """
        Dr. John Smith
        Professor of Computer Science
        Email: jsmith@mit.edu
        Research: Machine Learning, Artificial Intelligence
        
        Dr. Jane Doe  
        Associate Professor
        Email: jane@mit.edu
        Research: Computer Vision, Robotics
        """

        user_profile = "PhD student interested in Machine Learning and AI applications"

        prompt = f"""
        Extract professor information from this content and match with user profile.
        
        Content: {sample_content}
        User Profile: {user_profile}
        
        Output JSON array with professors having alignment_score >= 6.0:
        [
          {{
            "name": "Dr. John Smith",
            "email": "jsmith@mit.edu", 
            "research_interests": "Machine Learning, AI",
            "alignment_score": 9.0,
            "collaboration_potential": "Strong match for ML research"
          }}
        ]
        """

        print_colored("Making Stage 1 API request...", "96")
        start_time = time.time()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            timeout=30
        )

        api_time = time.time() - start_time
        result = response.choices[0].message.content

        print_colored(f"✅ API completed in {api_time:.2f}s", "92")
        print_colored(f"Response preview: {result[:100]}...", "96")

        # Try to parse JSON
        import json
        import re
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            professors_data = json.loads(json_match.group())
            print_colored(
                f"✅ Found {len(professors_data)} professors in response", "92")
            return True
        else:
            print_colored("⚠️ No valid JSON found in response", "93")
            return False

    except Exception as e:
        print_colored(f"❌ Stage 1 AI test failed: {e}", "91")
        return False


def run_complete_stage1_simulation():
    """Run a complete but simplified Stage 1 simulation."""
    print_colored("🎯 Running complete Stage 1 simulation...", "93")

    # Setup
    driver = setup_simple_driver()
    if not driver:
        return False

    try:
        university = "MIT"
        departments = ["Computer Science"]

        print_colored(f"Simulating: {university} - {departments[0]}", "96")

        # Step 1: Navigate to faculty page
        print_colored("Step 1: Navigate to faculty page...", "96")
        start_time = time.time()
        driver.get("https://www.csail.mit.edu/people")
        nav_time = time.time() - start_time
        print_colored(f"✅ Navigation completed in {nav_time:.2f}s", "92")

        # Step 2: Extract content
        print_colored("Step 2: Extract page content...", "96")
        content = driver.page_source
        content_length = len(content)
        print_colored(
            f"✅ Content extracted: {content_length} characters", "92")

        # Step 3: Process with AI (using shortened content)
        print_colored("Step 3: Process with AI...", "96")

        # Truncate content to avoid timeouts
        truncated_content = content[:3000] if len(content) > 3000 else content

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print_colored("❌ No API key for AI processing", "91")
            return False

        client = openai.OpenAI(api_key=api_key)

        prompt = f"""
        Extract professor information from this MIT CSAIL page content.
        
        Content: {truncated_content}
        
        Find professors and output as JSON array (max 3 professors):
        [
          {{
            "name": "Professor Name",
            "email": "email@mit.edu",
            "research_interests": "Research areas",
            "alignment_score": 8.0
          }}
        ]
        
        Only include professors with clear research information.
        """

        ai_start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            timeout=45
        )
        ai_time = time.time() - ai_start_time

        print_colored(f"✅ AI processing completed in {ai_time:.2f}s", "92")

        # Parse results
        result = response.choices[0].message.content
        import json
        import re

        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            professors_data = json.loads(json_match.group())
            print_colored(
                f"✅ Successfully extracted {len(professors_data)} professors", "92")

            for i, prof in enumerate(professors_data[:3]):  # Show first 3
                print_colored(f"  {i+1}. {prof.get('name', 'Unknown')}", "96")
                print_colored(
                    f"     Research: {prof.get('research_interests', 'N/A')[:50]}...", "96")

            total_time = time.time() - start_time
            print_colored(
                f"🎉 Complete simulation successful in {total_time:.2f}s total", "92")
            return True
        else:
            print_colored("❌ Failed to parse AI response", "91")
            return False

    except Exception as e:
        print_colored(f"❌ Simulation failed: {e}", "91")
        return False
    finally:
        if driver:
            driver.quit()


def diagnose_hanging_issue():
    """Diagnose why Stage 1 might be hanging."""
    print_colored("🔍 Diagnosing hanging issue...", "93")

    potential_issues = [
        ("WebDriver timeout", "Check if ChromeDriver is hanging on page loads"),
        ("Network connectivity", "Slow internet or blocked requests"),
        ("Google search rate limiting", "Too many search requests causing blocks"),
        ("OpenAI API timeout", "AI processing taking too long"),
        ("Memory issues", "Running out of RAM during processing"),
        ("Selenium compatibility", "ChromeDriver version mismatch")
    ]

    print_colored("\n🔍 Potential hanging causes:", "1;94")
    for i, (issue, description) in enumerate(potential_issues, 1):
        print_colored(f"{i}. {issue}: {description}", "96")

    print_colored("\n💡 Recommended fixes:", "1;94")
    fixes = [
        "Add timeouts to all web operations (20-30 seconds max)",
        "Use direct university URLs instead of Google search",
        "Process 1 university at a time with progress updates",
        "Add retry logic with exponential backoff",
        "Limit content size sent to OpenAI API",
        "Add memory monitoring and cleanup"
    ]

    for i, fix in enumerate(fixes, 1):
        print_colored(f"{i}. {fix}", "96")


def main():
    """Main test function."""
    print_colored("🧪 PhD Outreach - Stage 1 Isolated Test", "1;96")
    print_colored("="*60, "96")

    tests_passed = 0
    total_tests = 5

    # Test 1: OpenAI API
    print_colored("\n🧪 Test 1/5: OpenAI API Basic", "1;94")
    if test_simple_openai():
        tests_passed += 1

    # Test 2: WebDriver Setup
    print_colored("\n🧪 Test 2/5: WebDriver Setup", "1;94")
    driver = setup_simple_driver()
    if driver:
        tests_passed += 1

    # Test 3: Web Navigation
    print_colored("\n🧪 Test 3/5: Web Navigation", "1;94")
    if test_simple_web_navigation(driver):
        tests_passed += 1

    # Test 4: Faculty Page Discovery
    print_colored("\n🧪 Test 4/5: Faculty Page Discovery", "1;94")
    if test_faculty_page_discovery(driver):
        tests_passed += 1

    # Test 5: AI Processing
    print_colored("\n🧪 Test 5/5: AI Processing", "1;94")
    if test_minimal_stage1_process():
        tests_passed += 1

    # Cleanup
    if driver:
        driver.quit()

    # Results
    print_colored("\n" + "="*60, "96")
    print_colored(
        f"🎯 Test Results: {tests_passed}/{total_tests} passed", "1;96")

    if tests_passed == total_tests:
        print_colored(
            "✅ All tests passed! Running complete simulation...", "92")
        if run_complete_stage1_simulation():
            print_colored("\n🎉 Stage 1 simulation successful!", "1;92")
            print_colored("The hanging issue might be due to:", "96")
            print_colored("- Processing too many universities at once", "96")
            print_colored("- Network timeouts on specific websites", "96")
            print_colored("- Long AI processing times", "96")
        else:
            print_colored(
                "\n❌ Simulation failed - this is likely your issue", "91")
    else:
        print_colored(f"❌ {total_tests - tests_passed} tests failed", "91")
        diagnose_hanging_issue()

    # Recommendations
    print_colored("\n💡 Immediate Recommendations:", "1;94")
    print_colored("1. Process only 1 university at a time", "96")
    print_colored("2. Add 20-second timeouts to all operations", "96")
    print_colored("3. Use direct university URLs (avoid Google search)", "96")
    print_colored("4. Monitor progress in Streamlit interface", "96")
    print_colored("5. Add 'Stop' button functionality", "96")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n\n🛑 Test interrupted by user", "93")
    except Exception as e:
        print_colored(f"\n\n❌ Test error: {e}", "91")
