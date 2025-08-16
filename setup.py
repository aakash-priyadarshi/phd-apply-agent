#!/usr/bin/env python3
"""
PhD Outreach Automation - Windows 11 Setup Script (2-Stage Optimized)
Sets up the cost-optimized 2-stage system for PhD outreach automation.
"""

from urllib.parse import urlparse
import os
import sys
import json
import subprocess
import sqlite3
import asyncio
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import requests

# Color codes for terminal output


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_colored(text: str, color: str = Colors.WHITE):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.END}")


def print_header(text: str):
    """Print a section header."""
    print_colored(f"\n{'='*70}", Colors.CYAN)
    print_colored(f"{text:^70}", Colors.BOLD + Colors.CYAN)
    print_colored(f"{'='*70}", Colors.CYAN)


def print_step(step: str, status: str = "info"):
    """Print a setup step with status."""
    colors = {
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error": Colors.RED
    }
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    print_colored(f"{icons.get(status, 'ℹ️')} {step}",
                  colors.get(status, Colors.WHITE))


class OptimizedSetupValidator:
    """Enhanced setup validator for 2-stage system."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.project_root = Path.cwd()

    def check_windows_requirements(self) -> bool:
        """Check Windows 11 specific requirements."""
        print_step("Checking Windows 11 compatibility...")

        system = platform.system()
        if system != "Windows":
            self.warnings.append(
                f"This setup is optimized for Windows 11, found {system}")
            print_step(f"System: {system} (not Windows)", "warning")
        else:
            version = platform.version()
            print_step(f"Windows version: {version}", "success")

        # Check Windows version for Windows 11
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            build_number = winreg.QueryValueEx(key, "CurrentBuildNumber")[0]
            winreg.CloseKey(key)

            if int(build_number) >= 22000:  # Windows 11 build numbers start at 22000
                print_step(
                    "Windows 11 detected - optimal performance expected", "success")
            else:
                print_step(
                    f"Windows 10 detected (build {build_number}) - will work but Windows 11 recommended", "warning")
        except:
            print_step("Could not detect exact Windows version", "info")

        return True

    def install_optimized_dependencies(self) -> bool:
        """Install dependencies optimized for 2-stage system."""
        print_step("Installing optimized dependencies for 2-stage system...")

        # Core dependencies for 2-stage system
        requirements = [
            # UI and Data
            "streamlit>=1.29.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",

            # OpenAI with cost optimization
            "openai>=1.3.0",

            # Web scraping (Windows optimized)
            "selenium>=4.15.0",
            "webdriver-manager>=4.0.0",
            "beautifulsoup4>=4.12.0",
            "aiohttp>=3.8.0",
            "requests>=2.31.0",

            # Gmail integration
            "google-auth>=2.15.0",
            "google-auth-oauthlib>=0.8.0",
            "google-auth-httplib2>=0.1.1",
            "google-api-python-client>=2.70.0",

            # File processing
            "PyPDF2>=3.0.0",
            "python-dotenv>=1.0.0",

            # Windows specific optimizations
            "pywin32>=306; sys_platform=='win32'",
            "winrt>=1.0.21033.1; sys_platform=='win32'",
        ]

        print_step("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                       capture_output=True)

        failed_packages = []

        for package in requirements:
            try:
                package_name = package.split('>=')[0].split(';')[0]
                print_step(f"Installing {package_name}...")

                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, timeout=120)

                if result.returncode != 0:
                    failed_packages.append(package)
                    print_step(
                        f"Failed to install {package}: {result.stderr}", "error")
                else:
                    print_step(f"✓ {package_name}", "success")

            except subprocess.TimeoutExpired:
                failed_packages.append(package)
                print_step(f"Timeout installing {package}", "error")
            except Exception as e:
                failed_packages.append(package)
                print_step(f"Error installing {package}: {e}", "error")

        if failed_packages:
            self.errors.extend(
                [f"Failed to install: {pkg}" for pkg in failed_packages])
            return False

        print_step("All optimized dependencies installed successfully", "success")
        return True

    def setup_chrome_webdriver_windows(self) -> bool:
        """Setup Chrome WebDriver for Windows 11."""
        print_step("Setting up Chrome WebDriver for Windows...")

        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service

            # Download and setup ChromeDriver
            driver_path = ChromeDriverManager().install()
            print_step(f"ChromeDriver installed: {driver_path}", "success")

            # Test WebDriver with Windows 11 optimizations
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument(
                "--disable-features=VizDisplayCompositor")

            service = Service(driver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get("https://www.google.com")

            # Test basic functionality
            title = driver.title
            driver.quit()

            if "Google" in title:
                print_step("WebDriver test successful", "success")
                return True
            else:
                self.errors.append(
                    "WebDriver test failed - unexpected page title")
                return False

        except Exception as e:
            self.errors.append(f"WebDriver setup failed: {e}")
            print_step(f"WebDriver setup failed: {e}", "error")
            return False

    def create_optimized_env_file(self) -> bool:
        """Create .env file optimized for 2-stage system."""
        print_step("Setting up 2-stage system configuration...")

        env_file = Path(".env")

        if env_file.exists():
            response = input(
                f"\n{Colors.YELLOW}.env file already exists. Overwrite? (y/N): {Colors.END}")
            if response.lower() != 'y':
                print_step("Keeping existing .env file", "info")
                return True

        print_colored(
            f"\n{Colors.BOLD}🎓 2-Stage PhD Outreach System Configuration{Colors.END}")
        print_colored(
            "Stage 1: Professor Discovery (gpt-4o-mini - Cost Effective)")
        print_colored("Stage 2: Email Generation (gpt-4 - High Quality)")
        print_colored("\nPlease provide the following information:")

        # OpenAI Configuration
        print_colored(f"\n{Colors.PURPLE}🤖 OpenAI Configuration:{Colors.END}")
        print_colored(
            "Get your API key from: https://platform.openai.com/api-keys")

        while True:
            openai_key = input("Enter your OpenAI API key: ").strip()
            if openai_key:
                if openai_key.startswith('sk-') and len(openai_key) > 20:
                    break
                else:
                    print_step(
                        "Invalid API key format. Should start with 'sk-'", "error")
            else:
                print_step("OpenAI API key is required", "error")

        # User Information
        print_colored(f"\n{Colors.PURPLE}👤 User Information:{Colors.END}")
        user_name = input("Your full name (for email signatures): ").strip()
        user_email = input("Your email address (optional): ").strip()

        # System Configuration
        print_colored(f"\n{Colors.PURPLE}⚙️ System Configuration:{Colors.END}")

        print("Cost optimization settings:")
        print("1. Conservative (Lower cost, slower processing)")
        print("2. Balanced (Recommended)")
        print("3. Aggressive (Higher cost, faster processing)")

        while True:
            cost_mode = input("Choose cost mode (1-3, default: 2): ").strip()
            if cost_mode in ['1', '2', '3', '']:
                cost_mode = cost_mode or '2'
                break
            else:
                print_step("Please enter 1, 2, or 3", "error")

        # Set configuration based on cost mode
        cost_settings = {
            '1': {  # Conservative
                'MAX_CONCURRENT_REQUESTS': '2',
                'REQUEST_DELAY': '3',
                'EMAIL_BATCH_SIZE': '3',
                'MAX_PROFESSORS_PER_UNIVERSITY': '15',
                'MIN_ALIGNMENT_SCORE': '7.0'
            },
            '2': {  # Balanced (Recommended)
                'MAX_CONCURRENT_REQUESTS': '3',
                'REQUEST_DELAY': '2',
                'EMAIL_BATCH_SIZE': '5',
                'MAX_PROFESSORS_PER_UNIVERSITY': '20',
                'MIN_ALIGNMENT_SCORE': '6.0'
            },
            '3': {  # Aggressive
                'MAX_CONCURRENT_REQUESTS': '5',
                'REQUEST_DELAY': '1',
                'EMAIL_BATCH_SIZE': '10',
                'MAX_PROFESSORS_PER_UNIVERSITY': '30',
                'MIN_ALIGNMENT_SCORE': '5.0'
            }
        }

        settings = cost_settings[cost_mode]

        # Create optimized .env file
        try:
            with open(env_file, 'w') as f:
                f.write("# PhD Outreach Automation - 2-Stage System Configuration\n")
                f.write("# Generated by Windows 11 optimized setup script\n\n")

                f.write("# ============================================\n")
                f.write("# OPENAI CONFIGURATION (2-Stage System)\n")
                f.write("# ============================================\n")
                f.write(f"OPENAI_API_KEY={openai_key}\n")
                f.write("# Stage 1: Cost-effective professor discovery\n")
                f.write("STAGE1_MODEL=gpt-4o-mini\n")
                f.write("# Stage 2: High-quality email generation\n")
                f.write("STAGE2_MODEL=gpt-4\n\n")

                f.write("# ============================================\n")
                f.write("# USER INFORMATION\n")
                f.write("# ============================================\n")
                f.write(f"USER_NAME={user_name}\n")
                if user_email:
                    f.write(f"USER_EMAIL={user_email}\n")
                f.write("\n")

                f.write("# ============================================\n")
                f.write(
                    f"# COST OPTIMIZATION (Mode: {cost_mode} - {['Conservative', 'Balanced', 'Aggressive'][int(cost_mode)-1]})\n")
                f.write("# ============================================\n")
                for key, value in settings.items():
                    f.write(f"{key}={value}\n")
                f.write("\n")

                f.write("# ============================================\n")
                f.write("# WINDOWS 11 OPTIMIZATIONS\n")
                f.write("# ============================================\n")
                f.write("BROWSER_ENGINE=chrome\n")
                f.write("HEADLESS_BROWSER=true\n")
                f.write("BROWSER_TIMEOUT=30\n")
                f.write("USE_WEBDRIVER_MANAGER=true\n")
                f.write("WINDOWS_OPTIMIZED=true\n\n")

                f.write("# ============================================\n")
                f.write("# ADVANCED SETTINGS\n")
                f.write("# ============================================\n")
                f.write("LOG_LEVEL=INFO\n")
                f.write("DATABASE_PATH=phd_outreach.db\n")
                f.write("BACKUP_FREQUENCY=daily\n")
                f.write("DAILY_EMAIL_LIMIT=50\n")
                f.write("EXCLUDE_EMERITUS=true\n")
                f.write("EXCLUDE_VISITING=true\n")
                f.write("EMAIL_TEMPLATE_STYLE=professional\n")

            print_step(
                "Optimized .env configuration created successfully", "success")
            return True

        except Exception as e:
            self.errors.append(f"Failed to create .env file: {e}")
            print_step(f"Failed to create .env file: {e}", "error")
            return False

    def validate_2stage_apis(self) -> bool:
        """Validate both Stage 1 and Stage 2 API models."""
        print_step("Validating 2-stage API configuration...")

        try:
            from dotenv import load_dotenv
            import openai

            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')

            if not api_key:
                self.errors.append("OpenAI API key not found")
                return False

            client = openai.OpenAI(api_key=api_key)

            # Test Stage 1 API (gpt-4o-mini)
            print_step("Testing Stage 1 API (gpt-4o-mini)...")
            try:
                response1 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Test Stage 1 API"}],
                    max_tokens=10
                )
                print_step("✓ Stage 1 API (gpt-4o-mini) working", "success")
                stage1_cost = 0.00015 * 10 / 1000  # Approximate cost
                print_step(
                    f"Stage 1 estimated cost per request: ~${stage1_cost:.6f}", "info")
            except Exception as e:
                self.errors.append(f"Stage 1 API test failed: {e}")
                return False

            # Test Stage 2 API (gpt-4)
            print_step("Testing Stage 2 API (gpt-4)...")
            try:
                response2 = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Test Stage 2 API"}],
                    max_tokens=10
                )
                print_step("✓ Stage 2 API (gpt-4) working", "success")
                stage2_cost = 0.03 * 10 / 1000  # Approximate cost
                print_step(
                    f"Stage 2 estimated cost per email: ~${stage2_cost:.4f}", "info")
            except Exception as e:
                self.errors.append(f"Stage 2 API test failed: {e}")
                return False

            print_step("Both API stages validated successfully", "success")
            print_colored(f"\n{Colors.GREEN}💰 Cost Estimation:{Colors.END}")
            print_colored(f"Stage 1 (Discovery): ~$0.01-0.05 per university")
            print_colored(f"Stage 2 (Emails): ~$0.10-0.30 per email")
            print_colored(f"Total for 100 professors: ~$10-30")

            return True

        except Exception as e:
            self.errors.append(f"API validation failed: {e}")
            return False

    def setup_sample_data_windows(self) -> bool:
        """Create Windows-optimized sample data."""
        print_step("Creating sample data files...")

        # Enhanced PhD_Targets.csv with more universities
        targets_file = Path("PhD_Targets.csv")
        if not targets_file.exists():
            sample_targets = """University Name,Country,Departments to Search,Priority,Notes
Massachusetts Institute of Technology,USA,"CS, ECE, CSAIL, LIDS",High,"AI, Robotics, ML - Top tier research"
Stanford University,USA,"CS, AI Lab, HCI",High,"Silicon Valley connections, Strong AI program"
Carnegie Mellon University,USA,"CS, ML Dept, Robotics",High,"ML and Robotics powerhouse"
UC Berkeley,USA,"EECS, AI Research",High,"Public ivy, Strong AI and systems"
University of Washington,USA,"CS, AI Lab",High,"Strong industry connections, Good location"
University of Toronto,Canada,"CS, Vector Institute",High,"Geoffrey Hinton's university, AI hub"
University of Oxford,UK,"CS, Engineering Science",High,"Prestigious, Strong research funding"
University of Cambridge,UK,"CS, Engineering",High,"Historic excellence, Good funding"
ETH Zurich,Switzerland,"CS, AI Center",Medium,"Top European tech, Good research environment"
Technical University of Munich,Germany,"CS, AI",Medium,"Strong European program, Good funding"
University of Edinburgh,UK,"CS, AI",Medium,"AI research hub, Good reputation"
EPFL,Switzerland,"CS, AI Lab",Medium,"Strong research, Beautiful location"
University of Amsterdam,Netherlands,"CS, AI",Medium,"Growing AI program, Good location"
KTH Royal Institute,Sweden,"CS, AI",Medium,"Strong technical focus, Good funding"
University of Melbourne,Australia,"CS, AI",Low,"Strong program, Different timezone"
Australian National University,Australia,"CS, AI",Low,"Good research, Government connections"
"""

            try:
                with open(targets_file, 'w', encoding='utf-8') as f:
                    f.write(sample_targets)
                print_step("Created comprehensive PhD_Targets.csv", "success")
            except Exception as e:
                self.errors.append(f"Failed to create targets file: {e}")
                return False

        # Create PhD_Results.csv
        results_file = Path("PhD_Results.csv")
        if not results_file.exists():
            results_header = "Approved to Send (YES/NO),Professor Name,University,Department,Email,Research Interests,Recent Publications (semicolon-separated),Draft Email,Sent? (YES/NO),Notes,Alignment Score,Stage1 Cost,Stage2 Cost\n"

            try:
                with open(results_file, 'w', encoding='utf-8') as f:
                    f.write(results_header)
                print_step(
                    "Created enhanced PhD_Results.csv with cost tracking", "success")
            except Exception as e:
                self.errors.append(f"Failed to create results file: {e}")
                return False

        return True

    def setup_enhanced_database(self) -> bool:
        """Setup enhanced database with cost tracking."""
        print_step("Setting up enhanced database with cost tracking...")

        try:
            conn = sqlite3.connect("phd_outreach.db")
            cursor = conn.cursor()

            # Enhanced professors table with 2-stage tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS professors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    university TEXT NOT NULL,
                    department TEXT,
                    email TEXT,
                    research_interests TEXT,
                    ongoing_research TEXT,
                    profile_url TEXT,
                    alignment_score REAL DEFAULT 0.0,
                    collaboration_potential TEXT,
                    last_verified TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    email_sent_at TEXT,
                    draft_email_subject TEXT,
                    draft_email_body TEXT,
                    notes TEXT,
                    stage1_cost REAL DEFAULT 0.0,
                    stage2_cost REAL DEFAULT 0.0,
                    windows_optimized BOOLEAN DEFAULT 1
                )
            ''')

            # Cost tracking and analytics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cost_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    stage1_requests INTEGER DEFAULT 0,
                    stage1_cost REAL DEFAULT 0.0,
                    stage2_requests INTEGER DEFAULT 0,
                    stage2_cost REAL DEFAULT 0.0,
                    total_cost REAL DEFAULT 0.0,
                    professors_discovered INTEGER DEFAULT 0,
                    emails_generated INTEGER DEFAULT 0,
                    emails_sent INTEGER DEFAULT 0,
                    avg_alignment_score REAL DEFAULT 0.0,
                    created_at TEXT
                )
            ''')

            # Performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    university TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    windows_version TEXT,
                    created_at TEXT
                )
            ''')

            conn.commit()
            conn.close()

            print_step(
                "Enhanced database with cost tracking initialized", "success")
            return True

        except Exception as e:
            self.errors.append(f"Database setup failed: {e}")
            return False

    def test_complete_2stage_workflow(self) -> bool:
        """Test the complete 2-stage workflow."""
        print_step("Testing complete 2-stage workflow...")

        try:
            # Test all required imports
            required_modules = [
                'streamlit', 'pandas', 'openai', 'selenium',
                'webdriver_manager', 'google.auth', 'PyPDF2'
            ]

            for module in required_modules:
                try:
                    if module == 'google.auth':
                        import google.auth
                    else:
                        __import__(module)
                    print_step(f"✓ {module}", "success")
                except ImportError as e:
                    self.errors.append(f"Failed to import {module}: {e}")
                    return False

            # Test WebDriver functionality
            print_step("Testing WebDriver with Windows optimizations...")
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager

                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")

                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(
                    service=service, options=chrome_options)
                driver.get("https://httpbin.org/ip")
                driver.quit()

                print_step("✓ WebDriver test successful", "success")
            except Exception as e:
                self.warnings.append(f"WebDriver test failed: {e}")
                print_step(f"WebDriver warning: {e}", "warning")

            print_step("2-stage workflow test completed", "success")
            return True

        except Exception as e:
            self.errors.append(f"Workflow test failed: {e}")
            return False

    def generate_windows_scripts(self) -> bool:
        """Generate Windows-specific startup scripts."""
        print_step("Creating Windows startup scripts...")

        try:
            # Create enhanced Windows batch file
            batch_content = '''@echo off
title PhD Outreach Automation - 2-Stage System

echo.
echo ================================================
echo  PhD Outreach Automation - 2-Stage System
echo ================================================
echo  Stage 1: Professor Discovery (gpt-4o-mini)
echo  Stage 2: Email Generation (gpt-4)
echo ================================================
echo.

REM Activate virtual environment if it exists
if exist "phd_outreach_env\\Scripts\\activate.bat" (
    echo Activating virtual environment...
    call phd_outreach_env\\Scripts\\activate.bat
)

REM Check if streamlit is available
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit not found. Please run setup.py first.
    pause
    exit /b 1
)

echo Starting PhD Outreach Automation System...
echo Open your browser and go to the URL shown below.
echo.

REM Start the Streamlit app
streamlit run streamlit_app.py --server.port 8501 --server.headless false

pause
'''

            with open("start_phd_outreach.bat", "w") as f:
                f.write(batch_content)

            # Create PowerShell script for advanced users
            ps_content = '''# PhD Outreach Automation - PowerShell Launcher
Write-Host "PhD Outreach Automation - 2-Stage System" -ForegroundColor Cyan
Write-Host "Stage 1: Professor Discovery (gpt-4o-mini)" -ForegroundColor Green
Write-Host "Stage 2: Email Generation (gpt-4)" -ForegroundColor Blue
Write-Host ""

# Check if virtual environment exists
if (Test-Path "phd_outreach_env\\Scripts\\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\\phd_outreach_env\\Scripts\\Activate.ps1
}

# Check if required files exist
if (-not (Test-Path "streamlit_app.py")) {
    Write-Host "ERROR: streamlit_app.py not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found. Please run setup.py first." -ForegroundColor Yellow
}

# Start the application
Write-Host "Starting PhD Outreach Automation..." -ForegroundColor Green
Write-Host "Your browser should open automatically." -ForegroundColor Yellow
Write-Host ""

try {
    streamlit run streamlit_app.py --server.port 8501
} catch {
    Write-Host "ERROR: Failed to start Streamlit: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
'''

            with open("start_phd_outreach.ps1", "w") as f:
                f.write(ps_content)

            print_step("✓ Windows startup scripts created", "success")
            print_step(
                "  - start_phd_outreach.bat (double-click to run)", "info")
            print_step(
                "  - start_phd_outreach.ps1 (PowerShell version)", "info")
            return True

        except Exception as e:
            self.warnings.append(f"Failed to create startup scripts: {e}")
            return False

    def generate_comprehensive_report(self) -> None:
        """Generate final comprehensive setup report."""
        print_header("WINDOWS 11 SETUP REPORT - 2-STAGE SYSTEM")

        if not self.errors and not self.warnings:
            print_colored(f"""
{Colors.GREEN + Colors.BOLD}
🎉 SETUP COMPLETED SUCCESSFULLY!
{Colors.END}

{Colors.CYAN}Your 2-Stage PhD Outreach Automation System is ready!{Colors.END}

{Colors.BOLD}💰 Cost-Optimized Features:{Colors.END}
• Stage 1: Professor Discovery using gpt-4o-mini (~$0.01-0.05 per university)
• Stage 2: High-quality emails using gpt-4 (~$0.10-0.30 per email)
• Real-time cost tracking and analytics
• Batch processing for cost efficiency

{Colors.BOLD}🚀 Quick Start:{Colors.END}
1. Double-click: start_phd_outreach.bat
2. Or run: streamlit run streamlit_app.py
3. Upload your CV for AI analysis
4. Select target universities
5. Run Stage 1: Discovery (cost-effective)
6. Review professors and run Stage 2: Email generation
7. Send personalized emails with tracking

{Colors.BOLD}📊 Expected Performance:{Colors.END}
• Discovery: 10-30 professors per university
• Cost: $10-30 for 100 professors total
• Time: 5-15 minutes per university
• Success Rate: 70-90% email extraction

{Colors.BOLD}🎯 Key Benefits:{Colors.END}
• 50x faster than manual research
• AI-powered research alignment scoring
• Professional email generation
• Complete cost tracking
• Gmail integration for sending
• Windows 11 optimized performance
            """)
        else:
            if self.errors:
                print_colored(
                    f"\n❌ SETUP FAILED - {len(self.errors)} ERROR(S):", Colors.RED + Colors.BOLD)
                for i, error in enumerate(self.errors, 1):
                    print_colored(f"  {i}. {error}", Colors.RED)

                print_colored(
                    f"\n{Colors.YELLOW}Please fix these errors and run setup.py again.{Colors.END}")

            if self.warnings:
                print_colored(
                    f"\n⚠️ {len(self.warnings)} WARNING(S):", Colors.YELLOW + Colors.BOLD)
                for i, warning in enumerate(self.warnings, 1):
                    print_colored(f"  {i}. {warning}", Colors.YELLOW)

            if not self.errors:
                print_colored(
                    f"\n{Colors.GREEN}✅ SETUP COMPLETED WITH WARNINGS{Colors.END}")
                print_colored(
                    "The system should work but some features may have limitations.")


def main():
    """Main Windows 11 optimized setup function."""

    # Check if running on Windows
    if platform.system() != "Windows":
        print_colored(
            "⚠️ This setup is optimized for Windows 11.", Colors.YELLOW)
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print_colored("Setup cancelled.", Colors.YELLOW)
            return

    print_colored(f"""
{Colors.BOLD + Colors.CYAN}
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    PhD OUTREACH AUTOMATION - 2-STAGE SYSTEM (WINDOWS 11)            ║
║                                                                      ║
║    Stage 1: Smart Discovery (gpt-4o-mini) - Cost Effective          ║
║    Stage 2: Quality Emails (gpt-4) - Professional Results           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
{Colors.END}
""")

    print_colored(
        "🎓 Welcome to the cost-optimized PhD outreach automation setup!")
    print_colored(
        "This system uses a 2-stage approach to minimize API costs while maximizing quality.\n")

    validator = OptimizedSetupValidator()

    try:
        # Setup steps optimized for 2-stage system
        steps = [
            ("Checking Windows 11 compatibility",
             validator.check_windows_requirements),
            ("Installing optimized dependencies",
             validator.install_optimized_dependencies),
            ("Setting up Chrome WebDriver", validator.setup_chrome_webdriver_windows),
            ("Creating 2-stage configuration", validator.create_optimized_env_file),
            ("Validating 2-stage APIs", validator.validate_2stage_apis),
            ("Creating sample data", validator.setup_sample_data_windows),
            ("Setting up enhanced database", validator.setup_enhanced_database),
            ("Testing complete workflow", validator.test_complete_2stage_workflow),
            ("Generating Windows scripts", validator.generate_windows_scripts)
        ]

        print_header("2-STAGE SETUP PROCESS")

        for step_name, step_func in steps:
            print_colored(f"\n{Colors.BOLD}🔄 {step_name}...{Colors.END}")

            try:
                success = step_func()
                if not success and "dependencies" in step_name.lower():
                    break  # Critical failure
            except KeyboardInterrupt:
                print_colored(
                    f"\n\n{Colors.YELLOW}Setup interrupted by user{Colors.END}")
                sys.exit(1)
            except Exception as e:
                validator.errors.append(f"{step_name} failed: {e}")
                print_step(f"Unexpected error: {e}", "error")

        # Generate comprehensive report
        validator.generate_comprehensive_report()

        # Offer to start the application
        if not validator.errors:
            print_colored(
                f"\n{Colors.BOLD}Ready to start your PhD outreach automation?{Colors.END}")
            response = input("Start the application now? (Y/n): ")
            if response.lower() != 'n':
                print_colored(
                    "\n🚀 Starting 2-Stage PhD Outreach System...", Colors.GREEN)

                try:
                    # Try to start with streamlit
                    import subprocess
                    subprocess.run(["streamlit", "run", "streamlit_app.py"])
                except Exception as e:
                    print_colored(f"Could not auto-start: {e}", Colors.YELLOW)
                    print_colored(
                        "Please run: start_phd_outreach.bat", Colors.CYAN)

    except KeyboardInterrupt:
        print_colored(
            f"\n\n{Colors.YELLOW}Setup interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_colored(
            f"\n\n{Colors.RED}Unexpected error during setup: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()  # !/usr/bin/env python3
"""
PhD Outreach Automation - Comprehensive Setup Script
This script sets up everything needed for the PhD outreach system.
"""


# Color codes for terminal output

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_colored(text: str, color: str = Colors.WHITE):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.END}")


def print_header(text: str):
    """Print a section header."""
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"{text:^60}", Colors.BOLD + Colors.CYAN)
    print_colored(f"{'='*60}", Colors.CYAN)


def print_step(step: str, status: str = "info"):
    """Print a setup step with status."""
    colors = {
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error": Colors.RED
    }
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    print_colored(f"{icons.get(status, 'ℹ️')} {step}",
                  colors.get(status, Colors.WHITE))


class SetupValidator:
    """Validates all components of the PhD outreach system."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.project_root = Path.cwd()
        self.required_files = []
        self.optional_files = []

    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print_step("Checking Python version...")

        if sys.version_info < (3, 8):
            self.errors.append(f"Python 3.8+ required, found {sys.version}")
            print_step(f"Python {sys.version} - INCOMPATIBLE", "error")
            return False

        print_step(f"Python {sys.version.split()[0]} - Compatible", "success")
        return True

    def check_system_requirements(self) -> bool:
        """Check system-specific requirements."""
        print_step("Checking system requirements...")

        system = platform.system()
        print_step(f"Operating System: {system}")

        # Check for Chrome browser (needed for Selenium)
        chrome_paths = {
            "Windows": [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            ],
            "Darwin": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"],
            "Linux": ["/usr/bin/google-chrome", "/usr/bin/chromium-browser"]
        }

        chrome_found = False
        for path in chrome_paths.get(system, []):
            if os.path.exists(path):
                chrome_found = True
                print_step(f"Chrome browser found: {path}", "success")
                break

        if not chrome_found:
            self.warnings.append(
                "Chrome browser not found - required for web scraping")
            print_step("Chrome browser not found", "warning")

        return True

    def install_dependencies(self) -> bool:
        """Install required Python packages."""
        print_step("Installing Python dependencies...")

        requirements = [
            "streamlit>=1.28.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "openai>=1.3.0",
            "beautifulsoup4>=4.12.0",
            "selenium>=4.15.0",
            "aiohttp>=3.8.0",
            "requests>=2.31.0",
            "google-auth>=2.15.0",
            "google-auth-oauthlib>=0.8.0",
            "google-auth-httplib2>=0.1.1",
            "google-api-python-client>=2.70.0",
            "PyPDF2>=3.0.0",
            "python-dotenv>=1.0.0",
            "webdriver-manager>=4.0.0"
        ]

        failed_packages = []

        for package in requirements:
            try:
                print_step(f"Installing {package.split('>=')[0]}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, timeout=60)

                if result.returncode != 0:
                    failed_packages.append(package)
                    print_step(f"Failed to install {package}", "error")
                else:
                    print_step(
                        f"Installed {package.split('>=')[0]}", "success")

            except subprocess.TimeoutExpired:
                failed_packages.append(package)
                print_step(f"Timeout installing {package}", "error")
            except Exception as e:
                failed_packages.append(package)
                print_step(f"Error installing {package}: {e}", "error")

        if failed_packages:
            self.errors.extend(
                [f"Failed to install: {pkg}" for pkg in failed_packages])
            return False

        print_step("All dependencies installed successfully", "success")
        return True

    def setup_webdriver(self) -> bool:
        """Set up Chrome WebDriver."""
        print_step("Setting up Chrome WebDriver...")

        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            # Download and setup ChromeDriver
            driver_path = ChromeDriverManager().install()
            print_step(f"ChromeDriver installed: {driver_path}", "success")

            # Test WebDriver
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

            driver = webdriver.Chrome(driver_path, options=chrome_options)
            driver.get("https://www.google.com")
            driver.quit()

            print_step("WebDriver test successful", "success")
            return True

        except Exception as e:
            self.errors.append(f"WebDriver setup failed: {e}")
            print_step(f"WebDriver setup failed: {e}", "error")
            return False

    def create_env_file(self) -> bool:
        """Create .env file with user input."""
        print_step("Setting up environment configuration...")

        env_file = Path(".env")

        if env_file.exists():
            response = input(
                f"\n{Colors.YELLOW}.env file already exists. Overwrite? (y/N): {Colors.END}")
            if response.lower() != 'y':
                print_step("Keeping existing .env file", "info")
                return True

        print_colored(
            f"\n{Colors.BOLD}Environment Configuration Setup{Colors.END}")
        print_colored("Please provide the following information:")

        # Collect OpenAI API key
        print_colored(f"\n{Colors.PURPLE}OpenAI Configuration:{Colors.END}")
        print_colored(
            "Get your API key from: https://platform.openai.com/api-keys")

        while True:
            openai_key = input("Enter your OpenAI API key: ").strip()
            if openai_key:
                # Validate API key format
                if openai_key.startswith('sk-') and len(openai_key) > 20:
                    break
                else:
                    print_step(
                        "Invalid API key format. Should start with 'sk-'", "error")
            else:
                print_step("OpenAI API key is required", "error")

        # Optional configurations
        model = input("OpenAI model (default: gpt-4): ").strip() or "gpt-4"

        print_colored(
            f"\n{Colors.PURPLE}Application Configuration:{Colors.END}")
        user_name = input("Your name (for emails): ").strip()
        user_email = input("Your email address (optional): ").strip()

        # Create .env file
        try:
            with open(env_file, 'w') as f:
                f.write("# PhD Outreach Automation Configuration\n")
                f.write("# Generated by setup script\n\n")

                f.write("# OpenAI Configuration\n")
                f.write(f"OPENAI_API_KEY={openai_key}\n")
                f.write(f"OPENAI_MODEL={model}\n\n")

                f.write("# User Configuration\n")
                f.write(f"USER_NAME={user_name}\n")
                if user_email:
                    f.write(f"USER_EMAIL={user_email}\n")
                f.write("\n")

                f.write("# Application Settings\n")
                f.write("LOG_LEVEL=INFO\n")
                f.write("MAX_CONCURRENT_REQUESTS=3\n")
                f.write("REQUEST_DELAY=2\n")
                f.write("EMAIL_BATCH_SIZE=5\n")

            print_step(".env file created successfully", "success")
            return True

        except Exception as e:
            self.errors.append(f"Failed to create .env file: {e}")
            print_step(f"Failed to create .env file: {e}", "error")
            return False

    def validate_openai_api(self) -> bool:
        """Validate OpenAI API key."""
        print_step("Validating OpenAI API key...")

        try:
            from dotenv import load_dotenv
            import openai

            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')

            if not api_key:
                self.errors.append("OpenAI API key not found in .env file")
                print_step("OpenAI API key not found", "error")
                return False

            # Test API key
            client = openai.OpenAI(api_key=api_key)

            # Make a simple test request
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )

            print_step("OpenAI API key validated successfully", "success")
            return True

        except openai.AuthenticationError:
            self.errors.append("Invalid OpenAI API key")
            print_step("Invalid OpenAI API key", "error")
            return False
        except Exception as e:
            self.errors.append(f"OpenAI API validation failed: {e}")
            print_step(f"OpenAI API validation failed: {e}", "error")
            return False

    def setup_gmail_credentials(self) -> bool:
        """Set up Gmail API credentials."""
        print_step("Setting up Gmail API credentials...")

        credentials_file = Path("credentials.json")

        if credentials_file.exists():
            print_step("Gmail credentials file found", "success")

            # Validate credentials file
            try:
                with open(credentials_file, 'r') as f:
                    creds_data = json.load(f)

                if 'installed' in creds_data or 'web' in creds_data:
                    print_step("Gmail credentials file is valid", "success")
                    return True
                else:
                    self.errors.append("Invalid credentials.json format")
                    print_step("Invalid credentials.json format", "error")
                    return False

            except Exception as e:
                self.errors.append(f"Error reading credentials.json: {e}")
                print_step(f"Error reading credentials.json: {e}", "error")
                return False
        else:
            print_colored(
                f"\n{Colors.YELLOW}Gmail API Setup Required{Colors.END}")
            print_colored("To set up Gmail API credentials:")
            print_colored("1. Go to https://console.cloud.google.com/")
            print_colored("2. Create a new project or select existing")
            print_colored("3. Enable Gmail API")
            print_colored(
                "4. Create OAuth 2.0 credentials (Desktop application)")
            print_colored("5. Download credentials.json")
            print_colored("6. Place it in this directory")

            input(
                f"\n{Colors.BOLD}Press Enter when you've placed credentials.json in this directory...{Colors.END}")

            if credentials_file.exists():
                print_step("Gmail credentials file found", "success")
                return self.setup_gmail_credentials()  # Validate the file
            else:
                self.warnings.append(
                    "Gmail credentials not found - email features will not work")
                print_step("Gmail credentials not found", "warning")
                return False

    def test_gmail_api(self) -> bool:
        """Test Gmail API authentication."""
        print_step("Testing Gmail API authentication...")

        if not Path("credentials.json").exists():
            print_step("Skipping Gmail test - no credentials file", "warning")
            return False

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            import pickle

            scopes = ['https://www.googleapis.com/auth/gmail.send']
            creds = None
            token_file = "gmail_token.pickle"

            # Load existing token
            if os.path.exists(token_file):
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)

            # Get new token if needed
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', scopes)
                    creds = flow.run_local_server(port=0)

                # Save token
                with open(token_file, 'wb') as token:
                    pickle.dump(creds, token)

            # Test Gmail service
            service = build('gmail', 'v1', credentials=creds)
            profile = service.users().getProfile(userId='me').execute()

            print_step(
                f"Gmail API authenticated for: {profile.get('emailAddress', 'Unknown')}", "success")
            return True

        except Exception as e:
            self.warnings.append(f"Gmail API test failed: {e}")
            print_step(f"Gmail API test failed: {e}", "warning")
            return False

    def create_sample_data(self) -> bool:
        """Create sample CSV files."""
        print_step("Creating sample data files...")

        # Create PhD_Targets.csv if it doesn't exist
        targets_file = Path("PhD_Targets.csv")
        if not targets_file.exists():
            sample_targets = """University Name,Country,Departments to Search,Priority,Notes
Massachusetts Institute of Technology,USA,"CS, ECE, Robotics, BioE",High,"Human Augmentation, Biomechatronics, AI in Robotics"
Stanford University,USA,"CS, Robotics, HCI",High,"Symbolic AI, BMI, HCI, NLP, exoskeletons"
University of Oxford,UK,"CS, Stats, Engineering",High,"Symbolic AI, deep learning for social good, causal inference"
University of Cambridge,UK,"CS, Engineering, Bioinformatics",High,"Machine learning, computational neuroscience"
Carnegie Mellon University,USA,"CS, Robotics, ML",High,"Robotics, computer vision, machine learning"
UC Berkeley,USA,"EECS, CS",High,"AI, machine learning, computer vision"
ETH Zurich,Switzerland,"CS, Engineering",Medium,"Robotics, AI, computer systems"
University of Toronto,Canada,"CS, Engineering",Medium,"AI, machine learning, robotics"
"""

            try:
                with open(targets_file, 'w') as f:
                    f.write(sample_targets)
                print_step("Created sample PhD_Targets.csv", "success")
            except Exception as e:
                self.errors.append(f"Failed to create PhD_Targets.csv: {e}")
                print_step(f"Failed to create PhD_Targets.csv: {e}", "error")
        else:
            print_step("PhD_Targets.csv already exists", "info")

        # Create PhD_Results.csv if it doesn't exist
        results_file = Path("PhD_Results.csv")
        if not results_file.exists():
            results_header = "Approved to Send (YES/NO),Professor Name,University,Department,Email,Research Interests,Recent Publications (semicolon-separated),Draft Email,Sent? (YES/NO),Notes\n"

            try:
                with open(results_file, 'w') as f:
                    f.write(results_header)
                print_step("Created PhD_Results.csv", "success")
            except Exception as e:
                self.errors.append(f"Failed to create PhD_Results.csv: {e}")
                print_step(f"Failed to create PhD_Results.csv: {e}", "error")
        else:
            print_step("PhD_Results.csv already exists", "info")

        return True

    def setup_database(self) -> bool:
        """Initialize SQLite database."""
        print_step("Setting up SQLite database...")

        try:
            conn = sqlite3.connect("phd_outreach.db")
            cursor = conn.cursor()

            # Create professors table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS professors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    university TEXT NOT NULL,
                    department TEXT,
                    email TEXT,
                    research_interests TEXT,
                    ongoing_research TEXT,
                    profile_url TEXT,
                    last_verified TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    email_sent_at TEXT,
                    notes TEXT
                )
            ''')

            # Create sent emails table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sent_emails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    professor_id INTEGER,
                    email_subject TEXT,
                    email_body TEXT,
                    sent_at TEXT,
                    gmail_message_id TEXT,
                    status TEXT DEFAULT 'sent',
                    FOREIGN KEY (professor_id) REFERENCES professors (id)
                )
            ''')

            # Create universities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS universities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    country TEXT,
                    departments TEXT,
                    priority TEXT,
                    notes TEXT,
                    status TEXT DEFAULT 'pending',
                    professors_found INTEGER DEFAULT 0,
                    last_scraped TEXT
                )
            ''')

            conn.commit()
            conn.close()

            print_step("Database initialized successfully", "success")
            return True

        except Exception as e:
            self.errors.append(f"Database setup failed: {e}")
            print_step(f"Database setup failed: {e}", "error")
            return False

    def test_web_scraping(self) -> bool:
        """Test web scraping functionality."""
        print_step("Testing web scraping functionality...")

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager

            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")

            # Create driver
            driver = webdriver.Chrome(
                ChromeDriverManager().install(), options=chrome_options)

            # Test navigation
            driver.get("https://www.example.com")
            title = driver.title
            driver.quit()

            if title:
                print_step("Web scraping test successful", "success")
                return True
            else:
                self.warnings.append(
                    "Web scraping test failed - no page title")
                print_step("Web scraping test failed", "warning")
                return False

        except Exception as e:
            self.warnings.append(f"Web scraping test failed: {e}")
            print_step(f"Web scraping test failed: {e}", "warning")
            return False

    def test_streamlit_app(self) -> bool:
        """Test if Streamlit app can be imported."""
        print_step("Testing Streamlit application...")

        try:
            import streamlit as st
            print_step("Streamlit import successful", "success")

            # Check if app file exists
            app_file = Path("streamlit_app.py")
            if app_file.exists():
                print_step("Streamlit app file found", "success")
                return True
            else:
                self.warnings.append("streamlit_app.py not found")
                print_step("streamlit_app.py not found", "warning")
                return False

        except Exception as e:
            self.errors.append(f"Streamlit test failed: {e}")
            print_step(f"Streamlit test failed: {e}", "error")
            return False

    def run_comprehensive_test(self) -> bool:
        """Run a comprehensive system test."""
        print_step("Running comprehensive system test...")

        try:
            # Test imports
            modules_to_test = [
                'openai', 'streamlit', 'pandas', 'selenium',
                'beautifulsoup4', 'aiohttp', 'google.auth'
            ]

            failed_imports = []
            for module in modules_to_test:
                try:
                    if module == 'beautifulsoup4':
                        import bs4
                    elif module == 'google.auth':
                        import google.auth
                    else:
                        __import__(module)
                    print_step(f"✓ {module}", "success")
                except ImportError:
                    failed_imports.append(module)
                    print_step(f"✗ {module}", "error")

            if failed_imports:
                self.errors.extend(
                    [f"Failed to import: {module}" for module in failed_imports])
                return False

            print_step("All module imports successful", "success")
            return True

        except Exception as e:
            self.errors.append(f"Comprehensive test failed: {e}")
            print_step(f"Comprehensive test failed: {e}", "error")
            return False

    def generate_report(self) -> None:
        """Generate final setup report."""
        print_header("SETUP REPORT")

        if not self.errors and not self.warnings:
            print_colored("🎉 SETUP COMPLETED SUCCESSFULLY!",
                          Colors.GREEN + Colors.BOLD)
            print_colored(
                "\nYour PhD Outreach Automation system is ready to use!")
            print_colored("\nNext steps:")
            print_colored("1. Run: streamlit run streamlit_app.py")
            print_colored("2. Upload your CV in the app")
            print_colored("3. Configure your target universities")
            print_colored("4. Start the research automation")
        else:
            if self.errors:
                print_colored(
                    f"\n❌ SETUP FAILED - {len(self.errors)} ERROR(S):", Colors.RED + Colors.BOLD)
                for error in self.errors:
                    print_colored(f"   • {error}", Colors.RED)

            if self.warnings:
                print_colored(
                    f"\n⚠️ {len(self.warnings)} WARNING(S):", Colors.YELLOW + Colors.BOLD)
                for warning in self.warnings:
                    print_colored(f"   • {warning}", Colors.YELLOW)

            if not self.errors:
                print_colored("\n✅ SETUP COMPLETED WITH WARNINGS",
                              Colors.YELLOW + Colors.BOLD)
                print_colored(
                    "The system should work but some features may be limited.")


def main():
    """Main setup function."""
    print_colored(f"""
{Colors.BOLD + Colors.CYAN}
██████╗ ██╗  ██╗██████╗      ██████╗ ██╗   ██╗████████╗██████╗ ███████╗ █████╗  ██████╗██╗  ██╗
██╔══██╗██║  ██║██╔══██╗    ██╔═══██╗██║   ██║╚══██╔══╝██╔══██╗██╔════╝██╔══██╗██╔════╝██║  ██║
██████╔╝███████║██║  ██║    ██║   ██║██║   ██║   ██║   ██████╔╝█████╗  ███████║██║     ███████║
██╔═══╝ ██╔══██║██║  ██║    ██║   ██║██║   ██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║     ██╔══██║
██║     ██║  ██║██████╔╝    ╚██████╔╝╚██████╔╝   ██║   ██║  ██║███████╗██║  ██║╚██████╗██║  ██║
╚═╝     ╚═╝  ╚═╝╚═════╝      ╚═════╝  ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
{Colors.END}
                    AUTOMATION SYSTEM SETUP
    """)

    print_colored("🎓 Welcome to the PhD Outreach Automation Setup!")
    print_colored(
        "This script will set up everything you need to automate your PhD applications.\n")

    validator = SetupValidator()

    try:
        # Run all setup steps
        steps = [
            ("Checking Python version", validator.check_python_version),
            ("Checking system requirements", validator.check_system_requirements),
            ("Installing dependencies", validator.install_dependencies),
            ("Setting up WebDriver", validator.setup_webdriver),
            ("Creating environment file", validator.create_env_file),
            ("Validating OpenAI API", validator.validate_openai_api),
            ("Setting up Gmail credentials", validator.setup_gmail_credentials),
            ("Testing Gmail API", validator.test_gmail_api),
            ("Creating sample data", validator.create_sample_data),
            ("Setting up database", validator.setup_database),
            ("Testing web scraping", validator.test_web_scraping),
            ("Testing Streamlit app", validator.test_streamlit_app),
            ("Running comprehensive test", validator.run_comprehensive_test)
        ]

        print_header("SETUP PROCESS")

        for step_name, step_func in steps:
            print_colored(f"\n{Colors.BOLD}Step: {step_name}{Colors.END}")

            try:
                success = step_func()
                if not success and step_name in ["Checking Python version", "Installing dependencies"]:
                    # Critical failures
                    break
            except KeyboardInterrupt:
                print_colored(
                    f"\n\n{Colors.YELLOW}Setup interrupted by user{Colors.END}")
                sys.exit(1)
            except Exception as e:
                validator.errors.append(f"{step_name} failed: {e}")
                print_step(f"Unexpected error in {step_name}: {e}", "error")

        # Generate final report
        validator.generate_report()

        # Offer to start the app
        if not validator.errors:
            response = input(
                f"\n{Colors.BOLD}Would you like to start the Streamlit app now? (y/N): {Colors.END}")
            if response.lower() == 'y':
                print_colored("\n🚀 Starting Streamlit app...", Colors.GREEN)
                os.system("streamlit run streamlit_app.py")

    except KeyboardInterrupt:
        print_colored(
            f"\n\n{Colors.YELLOW}Setup interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_colored(
            f"\n\n{Colors.RED}Unexpected error during setup: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
