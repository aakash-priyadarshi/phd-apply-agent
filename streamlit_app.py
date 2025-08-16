#!/usr/bin/env python3
"""
PhD Outreach Automation - Optimized 2-Stage System (No WebDriver)
Stage 1: Extract & Match (gpt-4o-mini) - Cost-effective professor discovery
Stage 2: Email Drafting (gpt-4) - High-quality personalized emails
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os
import time
from pathlib import Path
import sqlite3
from typing import List, Dict, Optional, Any
import base64
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Web scraping imports (NO SELENIUM)
import requests
from bs4 import BeautifulSoup
import re

# Third-party imports
import openai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pickle
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding to handle emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phd_outreach.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Remove emojis from log messages for Windows compatibility


def clean_log_message(message: str) -> str:
    """Remove emojis from log messages to prevent encoding errors."""
    import re
    # Remove emojis and other Unicode symbols that cause Windows logging issues
    return re.sub(r'[^\x00-\x7F]+', '', message).strip()


@dataclass
class Professor:
    """Professor data structure with alignment scoring."""
    name: str
    university: str
    department: str
    email: str = ""
    research_interests: str = ""
    ongoing_research: str = ""
    profile_url: str = ""
    alignment_score: float = 0.0  # 1-10 alignment with user's research
    collaboration_potential: str = ""
    last_verified: str = ""
    status: str = "pending"  # pending, verified, email_drafted, email_sent, rejected
    created_at: str = ""
    email_sent_at: str = ""
    draft_email_subject: str = ""
    draft_email_body: str = ""
    notes: str = ""
    stage1_cost: float = 0.0  # Track API costs
    stage2_cost: float = 0.0


@dataclass
class University:
    """University target structure with progress tracking."""
    name: str
    country: str
    departments: str
    priority: str
    notes: str
    status: str = "pending"
    professors_found: int = 0
    professors_processed: int = 0
    total_stage1_cost: float = 0.0
    total_stage2_cost: float = 0.0
    last_scraped: str = ""


def safe_operation(func, *args, **kwargs):
    """Safe wrapper for operations that might fail."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Safe operation failed: {e}")
        return None


class APIManager:
    """Manages OpenAI API calls with cost tracking."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.total_cost = 0.0
        self.stage1_cost = 0.0
        self.stage2_cost = 0.0

        # Pricing per 1K tokens (as of 2024)
        self.pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06}
        }

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for API call."""
        if model not in self.pricing:
            return 0.0

        input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
        return input_cost + output_cost

    def stage1_extract_and_match_sync(self, page_content: str, user_research_profile: str) -> Dict[str, Any]:
        """Stage 1: Extract professor info and match with user research (gpt-4o-mini) - Sync version."""
        prompt = f"""
        Extract professor information from this faculty page content and match with user's research profile.
        
        Faculty Page Content:
        {page_content[:4000]}  # Limit content to control costs
        
        User's Research Profile:
        {user_research_profile}
        
        Extract up to 10 professors and for each provide:
        1. Name and basic info
        2. Research areas/interests
        3. Alignment score (1-10) with user's profile
        4. Brief reasoning for the score
        
        Output as JSON array:
        [
          {{
            "name": "Dr. John Smith",
            "email": "jsmith@university.edu",
            "research_interests": "Machine Learning, Computer Vision",
            "profile_url": "https://...",
            "alignment_score": 8.5,
            "collaboration_potential": "Strong alignment in ML and CV research"
          }}
        ]
        
        Only include professors with alignment_score >= 6.0
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )

            # Calculate cost
            cost = self.calculate_cost(
                "gpt-4o-mini",
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            self.stage1_cost += cost
            self.total_cost += cost

            content = response.choices[0].message.content

            # Parse JSON response
            try:
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    professors_data = json.loads(json_match.group())
                    return {
                        "professors": professors_data,
                        "cost": cost,
                        "success": True
                    }
                else:
                    return {"professors": [], "cost": cost, "success": False, "error": "No JSON found"}
            except json.JSONDecodeError as e:
                return {"professors": [], "cost": cost, "success": False, "error": f"JSON parse error: {e}"}

        except Exception as e:
            logger.error(f"Stage 1 API error: {e}")
            return {"professors": [], "cost": 0, "success": False, "error": str(e)}

    def stage2_draft_email_sync(self, professor: Professor, user_research_profile: str,
                                user_name: str) -> Dict[str, Any]:
        """Stage 2: Draft personalized email (gpt-4) - Sync version."""
        prompt = f"""
        Write a professional, personalized PhD application email for this professor.
        
        Professor Details:
        - Name: {professor.name}
        - University: {professor.university}
        - Department: {professor.department}
        - Research: {professor.research_interests}
        - Collaboration Potential: {professor.collaboration_potential}
        - Alignment Score: {professor.alignment_score}/10
        
        Applicant Profile:
        {user_research_profile}
        
        Applicant Name: {user_name}
        
        Requirements:
        1. Professional academic tone
        2. Specific reference to professor's research
        3. Clear research alignment explanation
        4. Mention attached CV
        5. Request for PhD opportunity discussion
        6. Keep email body under 180 words
        7. Subject line under 80 characters
        
        Output as JSON:
        {{
            "subject": "Compelling subject line mentioning specific research area",
            "body": "Professional email body with specific research connections",
            "key_points": ["alignment point 1", "alignment point 2"],
            "tone_analysis": "professional/enthusiastic/research-focused"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )

            # Calculate cost
            cost = self.calculate_cost(
                "gpt-4",
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            self.stage2_cost += cost
            self.total_cost += cost

            content = response.choices[0].message.content

            # Parse JSON response
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    email_data = json.loads(json_match.group())
                    email_data["cost"] = cost
                    email_data["success"] = True
                    return email_data
                else:
                    return {"success": False, "cost": cost, "error": "No JSON found"}
            except json.JSONDecodeError as e:
                return {"success": False, "cost": cost, "error": f"JSON parse error: {e}"}

        except Exception as e:
            logger.error(f"Stage 2 API error: {e}")
            return {"success": False, "cost": 0, "error": str(e)}


class DatabaseManager:
    """Enhanced database manager with cost tracking."""

    def __init__(self, db_path: str = "phd_outreach.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables with cost tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enhanced professors table
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
                stage2_cost REAL DEFAULT 0.0
            )
        ''')

        # Cost tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cost_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                stage1_cost REAL DEFAULT 0.0,
                stage2_cost REAL DEFAULT 0.0,
                total_cost REAL DEFAULT 0.0,
                professors_processed INTEGER DEFAULT 0,
                emails_generated INTEGER DEFAULT 0,
                created_at TEXT
            )
        ''')

        # University progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS university_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                university_name TEXT UNIQUE NOT NULL,
                status TEXT DEFAULT 'pending',
                professors_found INTEGER DEFAULT 0,
                professors_processed INTEGER DEFAULT 0,
                stage1_cost REAL DEFAULT 0.0,
                stage2_cost REAL DEFAULT 0.0,
                last_updated TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def add_professor(self, professor: Professor) -> int:
        """Add professor with enhanced data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO professors (
                name, university, department, email, research_interests,
                ongoing_research, profile_url, alignment_score, collaboration_potential,
                last_verified, status, created_at, notes, stage1_cost, stage2_cost,
                draft_email_subject, draft_email_body
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            professor.name, professor.university, professor.department, professor.email,
            professor.research_interests, professor.ongoing_research, professor.profile_url,
            professor.alignment_score, professor.collaboration_potential, professor.last_verified,
            professor.status, professor.created_at, professor.notes, professor.stage1_cost,
            professor.stage2_cost, professor.draft_email_subject, professor.draft_email_body
        ))

        professor_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return professor_id

    def update_cost_tracking(self, stage1_cost: float, stage2_cost: float,
                             professors_processed: int, emails_generated: int):
        """Update daily cost tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")

        # Check if record exists for today
        cursor.execute('SELECT id FROM cost_tracking WHERE date = ?', (today,))
        record = cursor.fetchone()

        if record:
            # Update existing record
            cursor.execute('''
                UPDATE cost_tracking 
                SET stage1_cost = stage1_cost + ?, stage2_cost = stage2_cost + ?,
                    total_cost = stage1_cost + stage2_cost + ? + ?,
                    professors_processed = professors_processed + ?,
                    emails_generated = emails_generated + ?
                WHERE date = ?
            ''', (stage1_cost, stage2_cost, stage1_cost, stage2_cost,
                  professors_processed, emails_generated, today))
        else:
            # Create new record
            cursor.execute('''
                INSERT INTO cost_tracking (
                    date, stage1_cost, stage2_cost, total_cost,
                    professors_processed, emails_generated, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (today, stage1_cost, stage2_cost, stage1_cost + stage2_cost,
                  professors_processed, emails_generated, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Today's costs
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute('SELECT * FROM cost_tracking WHERE date = ?', (today,))
        today_data = cursor.fetchone()

        # Total costs
        cursor.execute('''
            SELECT SUM(stage1_cost), SUM(stage2_cost), SUM(total_cost),
                   SUM(professors_processed), SUM(emails_generated)
            FROM cost_tracking
        ''')
        total_data = cursor.fetchone()

        conn.close()

        return {
            "today": {
                "stage1_cost": today_data[2] if today_data else 0.0,
                "stage2_cost": today_data[3] if today_data else 0.0,
                "total_cost": today_data[4] if today_data else 0.0,
                "professors_processed": today_data[5] if today_data else 0,
                "emails_generated": today_data[6] if today_data else 0
            },
            "total": {
                "stage1_cost": total_data[0] if total_data[0] else 0.0,
                "stage2_cost": total_data[1] if total_data[1] else 0.0,
                "total_cost": total_data[2] if total_data[2] else 0.0,
                "professors_processed": total_data[3] if total_data[3] else 0,
                "emails_generated": total_data[4] if total_data[4] else 0
            }
        }


class NoWebDriverScraper:
    """Pure HTTP scraper using requests + BeautifulSoup (NO SELENIUM)."""

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.session = requests.Session()

        # Set up headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def scrape_university_faculty_sync(self, university: str, departments: List[str],
                                       user_research_profile: str) -> List[Professor]:
        """Scrape university faculty using pure HTTP requests."""
        professors = []
        logger.info(f"Starting scrape for {university}")

        try:
            for dept in departments:
                # Find faculty pages
                faculty_pages = self.find_faculty_pages_sync(university, dept)
                logger.info(
                    f"Found {len(faculty_pages)} faculty pages for {dept}")

                # Limit to top 2 pages per department to control costs
                for page_url in faculty_pages[:2]:
                    try:
                        # Get page content using requests
                        response = self.session.get(page_url, timeout=20)
                        response.raise_for_status()

                        # Parse with BeautifulSoup
                        soup = BeautifulSoup(response.content, 'html.parser')
                        page_content = soup.get_text()

                        logger.info(
                            f"Scraped content from {page_url} - {len(page_content)} chars")

                        # Stage 1: Extract and match professors (cost-effective)
                        stage1_result = self.api_manager.stage1_extract_and_match_sync(
                            page_content, user_research_profile
                        )

                        if stage1_result["success"]:
                            for prof_data in stage1_result["professors"]:
                                professor = Professor(
                                    name=prof_data.get("name", ""),
                                    university=university,
                                    department=dept,
                                    email=prof_data.get("email", ""),
                                    research_interests=prof_data.get(
                                        "research_interests", ""),
                                    profile_url=prof_data.get(
                                        "profile_url", ""),
                                    alignment_score=prof_data.get(
                                        "alignment_score", 0.0),
                                    collaboration_potential=prof_data.get(
                                        "collaboration_potential", ""),
                                    status="verified" if prof_data.get(
                                        "alignment_score", 0) >= 6.0 else "pending",
                                    created_at=datetime.now().isoformat(),
                                    last_verified=datetime.now().isoformat(),
                                    stage1_cost=stage1_result["cost"] / len(
                                        stage1_result["professors"]) if stage1_result["professors"] else 0
                                )
                                professors.append(professor)

                            logger.info(
                                f"Stage 1 success: {len(stage1_result['professors'])} professors extracted")
                        else:
                            logger.warning(
                                f"Stage 1 failed: {stage1_result.get('error')}")

                        # Rate limiting
                        time.sleep(2)

                    except requests.RequestException as e:
                        logger.error(f"HTTP error scraping {page_url}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing {page_url}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error scraping {university}: {e}")

        logger.info(
            f"Completed scrape for {university}: {len(professors)} professors found")
        return professors

    def find_faculty_pages_sync(self, university: str, department: str) -> List[str]:
        """Find faculty pages using Google Search (API or headless browsing)."""
        faculty_urls = []

        try:
            # Option 1: Try Google Custom Search API first (if configured)
            faculty_urls = self.google_search_api(university, department)

            # Option 2: Fallback to headless Google search if API not available
            if not faculty_urls:
                faculty_urls = self.google_search_headless(
                    university, department)

            # Option 3: Fallback to direct university URLs if search fails
            if not faculty_urls:
                faculty_urls = self.direct_university_urls(
                    university, department)

        except Exception as e:
            logger.error(f"Error finding faculty pages: {e}")

        logger.info(
            f"Found {len(faculty_urls)} faculty pages for {university} {department}")
        return faculty_urls

    def google_search_api(self, university: str, department: str) -> List[str]:
        """Use Google Custom Search API for finding faculty pages."""
        faculty_urls = []

        try:
            # Check if Google Custom Search is configured
            google_api_key = os.getenv('GOOGLE_API_KEY')
            # Use your CSE ID as default
            google_cse_id = os.getenv('GOOGLE_CSE_ID', '2557e384b0f844ef9')

            if not google_api_key:
                logger.info(
                    "Google Custom Search API key not found, trying headless browsing...")
                return faculty_urls

            from googleapiclient.discovery import build

            # Build the service
            service = build("customsearch", "v1", developerKey=google_api_key)

            # Search terms optimized for faculty discovery
            search_queries = [
                f'"{university}" {department} faculty directory site:edu',
                f'"{university}" {department} professors site:edu',
                f'"{university}" {department} people faculty site:edu'
            ]

            for query in search_queries:
                try:
                    logger.info(f"Google API search: {query}")

                    # Execute the search
                    result = service.cse().list(
                        q=query,
                        cx=google_cse_id,
                        num=10
                    ).execute()

                    # Extract URLs from results
                    if 'items' in result:
                        for item in result['items']:
                            url = item['link']
                            title = item.get('title', '')

                            logger.info(f"Found result: {title} - {url}")

                            # Filter for faculty-related URLs
                            if any(keyword in url.lower() for keyword in ['faculty', 'people', 'staff', 'directory', 'professors']):
                                # Verify it's from the university domain
                                if self.is_university_domain(url, university):
                                    faculty_urls.append(url)
                                    logger.info(f"Added faculty URL: {url}")

                    if faculty_urls:
                        break  # Found results, no need to continue

                    time.sleep(0.1)  # Small delay between API calls

                except Exception as e:
                    logger.debug(f"Google API search error for {query}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Google Custom Search API error: {e}")

        logger.info(f"Google API found {len(faculty_urls)} URLs")
        return list(set(faculty_urls))  # Remove duplicates

    def google_search_headless(self, university: str, department: str) -> List[str]:
        """Use headless browsing for Google Search (cost-free alternative)."""
        faculty_urls = []

        try:
            # Import selenium only if needed
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            # Setup headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

            # Create driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            try:
                # Search terms for Google
                search_queries = [
                    f'"{university}" {department} faculty directory site:edu',
                    f'"{university}" {department} professors site:edu'
                ]

                for query in search_queries:
                    try:
                        # Navigate to Google
                        google_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                        driver.get(google_url)
                        time.sleep(2)

                        # Find search result links
                        search_results = driver.find_elements(
                            By.CSS_SELECTOR, "a[href]")

                        for result in search_results[:15]:  # Top 15 results
                            try:
                                href = result.get_attribute("href")
                                if href and href.startswith("http"):
                                    # Filter for faculty pages
                                    if any(keyword in href.lower() for keyword in ['faculty', 'people', 'staff', 'directory', 'professors']):
                                        # Verify university domain
                                        if self.is_university_domain(href, university):
                                            faculty_urls.append(href)
                            except:
                                continue

                        if faculty_urls:
                            break  # Found results

                        time.sleep(1)

                    except Exception as e:
                        logger.debug(
                            f"Google headless search error for {query}: {e}")
                        continue

            finally:
                driver.quit()

        except ImportError:
            logger.info(
                "Selenium not available for headless browsing, using direct URLs...")
        except Exception as e:
            logger.debug(f"Headless Google search error: {e}")

        return list(set(faculty_urls))

    def is_university_domain(self, url: str, university: str) -> bool:
        """Check if URL belongs to the university domain."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()

            # University domain mappings
            university_domains = {
                "Massachusetts Institute of Technology": ["mit.edu"],
                "Stanford University": ["stanford.edu"],
                "University of Oxford": ["ox.ac.uk", "oxford.ac.uk"],
                "University of Cambridge": ["cam.ac.uk", "cambridge.ac.uk"],
                "Carnegie Mellon University": ["cmu.edu"],
                "UC Berkeley": ["berkeley.edu"],
                "ETH Zurich": ["ethz.ch"],
                "University of Toronto": ["utoronto.ca"]
            }

            allowed_domains = university_domains.get(university, [])
            if not allowed_domains:
                # Fallback: extract domain from university name
                university_clean = university.lower().replace(" ", "").replace(
                    "university", "").replace("institute", "").replace("technology", "")
                return university_clean in domain

            return any(allowed_domain in domain for allowed_domain in allowed_domains)

        except Exception:
            return False

    def direct_university_urls(self, university: str, department: str) -> List[str]:
        """Fallback to direct university URLs when search fails."""
        faculty_urls = []

        # University-specific faculty page mappings (your existing code)
        university_mappings = {
            "Massachusetts Institute of Technology": {
                "cs_urls": ["https://www.eecs.mit.edu/people/faculty", "https://www.csail.mit.edu/people"],
                "ece_urls": ["https://www.eecs.mit.edu/people/faculty"],
                "robotics_urls": ["https://www.csail.mit.edu/research/robotics"],
                "bioe_urls": ["https://be.mit.edu/directory/faculty"]
            },
            "Stanford University": {
                "cs_urls": ["https://cs.stanford.edu/people/faculty"],
                "robotics_urls": ["https://ai.stanford.edu/people/"],
                "hci_urls": ["https://hci.stanford.edu/people/"]
            },
            "University of Oxford": {
                "cs_urls": ["https://www.cs.ox.ac.uk/people/faculty.html"],
                "stats_urls": ["https://www.stats.ox.ac.uk/people/"],
                "engineering_urls": ["https://eng.ox.ac.uk/people/"]
            },
            "University of Cambridge": {
                "cs_urls": ["https://www.cst.cam.ac.uk/people/academic-staff"],
                "engineering_urls": ["https://www.eng.cam.ac.uk/profiles/"]
            },
            "Carnegie Mellon University": {
                "cs_urls": ["https://www.cs.cmu.edu/directory/faculty"],
                "robotics_urls": ["https://www.ri.cmu.edu/faculty/"],
                "ml_urls": ["https://www.ml.cmu.edu/people/faculty.html"]
            },
            "UC Berkeley": {
                "eecs_urls": ["https://eecs.berkeley.edu/faculty"],
                "cs_urls": ["https://eecs.berkeley.edu/faculty"]
            },
            "ETH Zurich": {
                "cs_urls": ["https://inf.ethz.ch/people/faculty.html"]
            },
            "University of Toronto": {
                "cs_urls": ["https://web.cs.toronto.edu/people/faculty"],
                "engineering_urls": ["https://www.engineering.utoronto.ca/faculty-staff/"]
            }
        }

        university_info = university_mappings.get(university, {})
        dept_lower = department.lower()
        dept_urls = university_info.get(f"{dept_lower}_urls", [])

        # Test each URL
        for url in dept_urls[:2]:  # Limit to top 2
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    faculty_urls.append(url)
            except:
                continue

        return faculty_urls

    def close_session(self):
        """Close the requests session."""
        if self.session:
            self.session.close()


class ResearchOrchestrator:
    """Main orchestrator for the 2-stage research process."""

    def __init__(self, openai_api_key: str, gmail_credentials: str):
        self.api_manager = APIManager(openai_api_key)
        self.web_scraper = NoWebDriverScraper(self.api_manager)
        self.db = DatabaseManager()

        # Gmail setup
        try:
            from gmail_manager import GmailManager
            self.gmail_manager = GmailManager(gmail_credentials)
        except:
            self.gmail_manager = None

        self.is_running = False
        self.progress_messages = []

    def add_progress_message(self, message: str):
        """Add progress message for UI."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.progress_messages.append(formatted_message)
        if len(self.progress_messages) > 20:
            self.progress_messages = self.progress_messages[-20:]
        # Log without emojis to prevent encoding errors
        logger.info(clean_log_message(message))

    def run_stage1_research_sync(self, universities: List[University], user_research_profile: str):
        """Run Stage 1: Extract and match professors (cost-effective) - Sync version."""
        self.is_running = True
        self.add_progress_message(
            "🚀 Starting Stage 1: Professor Discovery & Matching...")

        try:
            for i, university in enumerate(universities):
                if not self.is_running:
                    break

                self.add_progress_message(
                    f"🔍 Stage 1: Researching {university.name} ({i+1}/{len(universities)})...")

                departments = [dept.strip()
                               for dept in university.departments.split(',')]
                professors = self.web_scraper.scrape_university_faculty_sync(
                    university.name, departments, user_research_profile
                )

                # Save professors to database
                professor_ids = []
                for professor in professors:
                    prof_id = safe_operation(self.db.add_professor, professor)
                    if prof_id:
                        professor_ids.append(prof_id)

                # Update cost tracking
                stage1_cost = sum(p.stage1_cost for p in professors)
                safe_operation(self.db.update_cost_tracking,
                               stage1_cost, 0.0, len(professors), 0)

                self.add_progress_message(
                    f"✅ Stage 1 Complete: {university.name} - {len(professors)} professors found (Cost: ${stage1_cost:.4f})"
                )

                time.sleep(3)  # Rate limiting between universities

        except Exception as e:
            logger.error(f"Stage 1 error: {e}")
            self.add_progress_message(f"❌ Stage 1 error: {e}")
        finally:
            self.is_running = False
            self.add_progress_message(
                "✅ Stage 1 Complete: Professor discovery finished!")

    def run_stage2_email_generation_sync(self, user_research_profile: str, user_name: str):
        """Run Stage 2: Generate high-quality personalized emails - Sync version."""
        self.add_progress_message("📧 Starting Stage 2: Email Generation...")

        try:
            # Get verified professors who need emails
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM professors 
                WHERE status = 'verified' AND alignment_score >= 6.0 
                AND (draft_email_subject IS NULL OR draft_email_subject = '')
                ORDER BY alignment_score DESC
            ''')

            columns = [description[0] for description in cursor.description]
            professor_rows = cursor.fetchall()
            conn.close()

            if not professor_rows:
                self.add_progress_message(
                    "ℹ️ No professors need email generation")
                return

            total_professors = len(professor_rows)
            self.add_progress_message(
                f"📧 Generating emails for {total_professors} professors...")

            emails_generated = 0
            total_stage2_cost = 0.0

            for i, row in enumerate(professor_rows):
                if not self.is_running:
                    break

                professor_data = dict(zip(columns, row))
                professor = Professor(
                    **{k: v for k, v in professor_data.items() if k != 'id'})

                self.add_progress_message(
                    f"✍️ Generating email for {professor.name} (Score: {professor.alignment_score:.1f}) [{i+1}/{total_professors}]"
                )

                # Stage 2: Generate personalized email
                email_result = self.api_manager.stage2_draft_email_sync(
                    professor, user_research_profile, user_name
                )

                if email_result["success"]:
                    # Update professor with email draft
                    conn = sqlite3.connect(self.db.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE professors 
                        SET draft_email_subject = ?, draft_email_body = ?, 
                            stage2_cost = ?, status = 'email_drafted'
                        WHERE id = ?
                    ''', (
                        email_result["subject"], email_result["body"],
                        email_result["cost"], professor_data['id']
                    ))
                    conn.commit()
                    conn.close()

                    emails_generated += 1
                    total_stage2_cost += email_result["cost"]

                    self.add_progress_message(
                        f"✅ Email generated for {professor.name} (Cost: ${email_result['cost']:.4f})")
                else:
                    self.add_progress_message(
                        f"❌ Failed to generate email for {professor.name}")

                time.sleep(1)  # Rate limiting

            # Update cost tracking
            safe_operation(self.db.update_cost_tracking, 0.0,
                           total_stage2_cost, 0, emails_generated)

            self.add_progress_message(
                f"✅ Stage 2 Complete: {emails_generated} emails generated (Total cost: ${total_stage2_cost:.4f})"
            )

        except Exception as e:
            logger.error(f"Stage 2 error: {e}")
            self.add_progress_message(f"❌ Stage 2 error: {e}")

    def generate_single_email_sync(self, professor_id: int, user_name: str, user_research_profile: str) -> bool:
        """Generate email for a single professor - Sync version."""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM professors WHERE id = ?', (professor_id,))

            columns = [description[0] for description in cursor.description]
            row = cursor.fetchone()
            conn.close()

            if row:
                professor_data = dict(zip(columns, row))
                professor = Professor(
                    **{k: v for k, v in professor_data.items() if k != 'id'})

                # Generate email
                email_result = self.api_manager.stage2_draft_email_sync(
                    professor, user_research_profile, user_name
                )

                if email_result["success"]:
                    # Update database
                    conn = sqlite3.connect(self.db.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE professors 
                        SET draft_email_subject = ?, draft_email_body = ?, 
                            stage2_cost = ?, status = 'email_drafted'
                        WHERE id = ?
                    ''', (
                        email_result["subject"], email_result["body"],
                        email_result["cost"], professor_id
                    ))
                    conn.commit()
                    conn.close()

                    self.add_progress_message(
                        f"✅ Email generated for {professor.name}")
                    return True
                else:
                    self.add_progress_message(
                        f"❌ Failed to generate email: {email_result.get('error', 'Unknown error')}")
                    return False
            else:
                self.add_progress_message("❌ Professor not found")
                return False

        except Exception as e:
            logger.error(
                f"Error generating email for professor {professor_id}: {e}")
            self.add_progress_message(f"❌ Error generating email: {e}")
            return False

    def stop_research(self):
        """Stop the research pipeline."""
        self.is_running = False
        if self.web_scraper:
            self.web_scraper.close_session()


# Synchronous wrapper functions for Streamlit compatibility
def run_stage1_sync(universities: List[University], user_research_profile: str, orchestrator: ResearchOrchestrator):
    """Synchronous wrapper for Stage 1 research."""
    return orchestrator.run_stage1_research_sync(universities, user_research_profile)


def run_stage2_sync(user_research_profile: str, user_name: str, orchestrator: ResearchOrchestrator):
    """Synchronous wrapper for Stage 2 email generation."""
    return orchestrator.run_stage2_email_generation_sync(user_research_profile, user_name)


def generate_single_email_sync(professor_id: int, user_name: str, orchestrator: ResearchOrchestrator, user_research_profile: str) -> bool:
    """Synchronous wrapper for single email generation."""
    return orchestrator.generate_single_email_sync(professor_id, user_name, user_research_profile)


# Streamlit UI
def main():
    st.set_page_config(
        page_title="PhD Outreach Automation - 2-Stage System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for Windows 11 styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #0078d4, #005a9e);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cost-card {
        background: #f8f9fa;
        border-left: 4px solid #0078d4;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stage-indicator {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
        color: #1565c0 !important;
    }
    .progress-message {
        background: #f8f9fa;
        padding: 0.3rem;
        border-radius: 3px;
        margin: 0.2rem 0;
        font-family: monospace;
        font-size: 0.8rem;
        color: #333 !important;
        border: 1px solid #dee2e6;
    }
    .professor-status {
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-verified { background: #d4edda; color: #155724; }
    .status-email-drafted { background: #d1ecf1; color: #0c5460; }
    .status-pending { background: #fff3cd; color: #856404; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header"><h1>🎓 PhD Outreach Automation - 2-Stage System</h1><p>Stage 1: Smart Discovery (gpt-4o-mini) • Stage 2: Quality Emails (gpt-4)</p></div>', unsafe_allow_html=True)

    # Initialize session state
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'research_profile' not in st.session_state:
        st.session_state.research_profile = ""
    if 'cv_uploaded' not in st.session_state:
        st.session_state.cv_uploaded = False
    if 'progress_messages' not in st.session_state:
        st.session_state.progress_messages = []

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Keys
        openai_key = st.text_input(
            "OpenAI API Key", type="password", value=os.getenv('OPENAI_API_KEY', ''))

        if openai_key:
            if st.session_state.orchestrator is None:
                try:
                    st.session_state.orchestrator = ResearchOrchestrator(
                        openai_key, "credentials.json")
                    st.success("✅ API connected!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {e}")

        st.markdown("---")

        # Cost tracking
        st.header("💰 Cost Tracking")
        if st.session_state.orchestrator:
            try:
                cost_summary = st.session_state.orchestrator.db.get_cost_summary()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Today's Cost",
                              f"${cost_summary['today']['total_cost']:.4f}")
                    st.caption(
                        f"Stage 1: ${cost_summary['today']['stage1_cost']:.4f}")
                    st.caption(
                        f"Stage 2: ${cost_summary['today']['stage2_cost']:.4f}")

                with col2:
                    st.metric("Total Cost",
                              f"${cost_summary['total']['total_cost']:.4f}")
                    st.caption(
                        f"Professors: {cost_summary['total']['professors_processed']}")
                    st.caption(
                        f"Emails: {cost_summary['total']['emails_generated']}")
            except Exception as e:
                st.error(f"Error loading cost data: {e}")

        st.markdown("---")

        # CV Upload
        st.header("📄 CV Upload")
        cv_file = st.file_uploader("Upload your CV (PDF)", type=['pdf'])

        if cv_file and st.session_state.orchestrator:
            if st.button("📊 Analyze CV"):
                with st.spinner("Analyzing CV with AI..."):
                    try:
                        pdf_reader = PyPDF2.PdfReader(BytesIO(cv_file.read()))
                        cv_text = ""
                        for page in pdf_reader.pages:
                            cv_text += page.extract_text() + "\n"

                        # Save CV file
                        with open("uploaded_cv.pdf", "wb") as f:
                            cv_file.seek(0)
                            f.write(cv_file.read())

                        # Generate research profile using synchronous call
                        research_profile_prompt = f"""
                        Analyze this CV and create a concise research profile for PhD applications:
                        
                        {cv_text[:3000]}
                        
                        Extract:
                        1. Primary research interests and expertise
                        2. Technical skills and methodologies
                        3. Key achievements and publications
                        4. Research goals and focus areas
                        
                        Create a 150-word research profile that can be used for professor matching.
                        """

                        # Use synchronous OpenAI call for Streamlit compatibility
                        response = st.session_state.orchestrator.api_manager.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "user", "content": research_profile_prompt}],
                            temperature=0.3,
                            max_tokens=500
                        )

                        st.session_state.research_profile = response.choices[0].message.content
                        st.session_state.cv_uploaded = True
                        st.success("✅ CV analyzed successfully!")

                        # Show research profile
                        st.text_area("Your Research Profile:",
                                     st.session_state.research_profile, height=100)

                    except Exception as e:
                        st.error(f"Error analyzing CV: {e}")

        # Google Custom Search API (using your existing setup)
        st.markdown("---")
        st.header("🔍 Google Search Configuration")

        st.info("🎯 **Your Custom Search Engine ID**: `2557e384b0f844ef9` (detected)")

        google_api_key = st.text_input(
            "Google API Key", type="password", value=os.getenv('GOOGLE_API_KEY', ''),
            help="Get API key from Google Cloud Console > APIs & Services > Credentials (same project as your Gmail credentials)")

        if google_api_key:
            st.success("✅ Google Custom Search API configured!")
            st.caption(
                "Will use Google Search API for better faculty discovery")
            # Auto-save to environment for this session
            os.environ['GOOGLE_API_KEY'] = google_api_key
            os.environ['GOOGLE_CSE_ID'] = '2557e384b0f844ef9'
        else:
            st.warning(
                "⚡ Add your Google API Key to enable powerful Google Search")
            with st.expander("📖 How to get Google API Key (2 minutes)"):
                st.markdown("""
                **Quick Setup - Same Project as Gmail:**
                
                1. **Go to Google Cloud Console**: [console.cloud.google.com](https://console.cloud.google.com/)
                2. **Select your project**: `jarvis-383903` (same as Gmail)
                3. **Enable API**: 
                   - Go to "APIs & Services" > "Library"
                   - Search for "Custom Search API"
                   - Click "Enable"
                4. **Create API Key**:
                   - Go to "APIs & Services" > "Credentials" 
                   - Click "+ CREATE CREDENTIALS" > "API Key"
                   - Copy the API key
                5. **Paste it above** ⬆️
                
                **Cost**: ~$5 per 1,000 searches (very affordable)
                **Alternative**: Leave blank to use free headless browsing (slower)
                """)

        # User settings
        st.markdown("---")
        st.header("👤 User Settings")
        user_name = st.text_input(
            "Your Name", value=os.getenv('USER_NAME', ''))
        user_email = st.text_input(
            "Your Email", value=os.getenv('USER_EMAIL', ''))

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🎯 Target Universities")

        # Load and display target universities
        universities_to_research = []

        if os.path.exists("PhD_Targets.csv"):
            try:
                df_targets = pd.read_csv("PhD_Targets.csv")

                # University selection and management
                for idx, row in df_targets.iterrows():
                    with st.expander(f"{row['University Name']} ({row['Priority']} Priority)"):
                        col_a, col_b, col_c = st.columns([2, 1, 1])

                        with col_a:
                            st.write(f"**Country:** {row['Country']}")
                            st.write(
                                f"**Departments:** {row['Departments to Search']}")
                            st.write(f"**Notes:** {row['Notes']}")

                        with col_b:
                            include = st.checkbox(
                                "Include", key=f"include_{idx}", value=True)

                        with col_c:
                            if st.button("🗑️ Remove", key=f"remove_{idx}"):
                                df_targets = df_targets.drop(idx)
                                df_targets.to_csv(
                                    "PhD_Targets.csv", index=False)
                                st.rerun()

                        if include:
                            universities_to_research.append(University(
                                name=row['University Name'],
                                country=row['Country'],
                                departments=row['Departments to Search'],
                                priority=row['Priority'],
                                notes=row['Notes']
                            ))
            except Exception as e:
                st.error(f"Error loading universities: {e}")
        else:
            st.warning(
                "PhD_Targets.csv not found. Please add universities below.")

        # Add new university
        st.subheader("➕ Add New University")
        with st.form("add_university"):
            new_name = st.text_input("University Name")
            new_country = st.text_input("Country")
            new_departments = st.text_input("Departments (comma-separated)")
            new_priority = st.selectbox("Priority", ["High", "Medium", "Low"])
            new_notes = st.text_area("Notes")

            if st.form_submit_button("Add University"):
                if new_name and new_country and new_departments:
                    try:
                        new_row = pd.DataFrame([{
                            'University Name': new_name,
                            'Country': new_country,
                            'Departments to Search': new_departments,
                            'Priority': new_priority,
                            'Notes': new_notes
                        }])

                        if os.path.exists("PhD_Targets.csv"):
                            df_targets = pd.read_csv("PhD_Targets.csv")
                            df_targets = pd.concat(
                                [df_targets, new_row], ignore_index=True)
                        else:
                            df_targets = new_row

                        df_targets.to_csv("PhD_Targets.csv", index=False)
                        st.success("✅ University added!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding university: {e}")
                else:
                    st.error("Please fill all required fields")

        # 2-Stage Research Controls
        st.markdown("---")
        st.header("🚀 2-Stage Research Pipeline")

        if st.session_state.orchestrator and st.session_state.cv_uploaded:

            # Stage indicators
            st.markdown(
                '<div class="stage-indicator"><strong>Stage 1:</strong> Professor Discovery &amp; Matching (gpt-4o-mini - Cost Effective)</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="stage-indicator"><strong>Stage 2:</strong> Personalized Email Generation (gpt-4 - High Quality)</div>',
                unsafe_allow_html=True)

            # Control buttons
            col_stage1, col_stage2, col_stop = st.columns(3)

            with col_stage1:
                if st.button("🔍 Run Stage 1", type="primary", help="Find and match professors (low cost)"):
                    if universities_to_research:
                        with st.spinner("Running Stage 1: Professor Discovery..."):
                            try:
                                run_stage1_sync(
                                    universities_to_research, st.session_state.research_profile, st.session_state.orchestrator)
                                st.success("✅ Stage 1 completed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Stage 1 failed: {e}")
                                logger.error(f"Stage 1 error: {e}")
                    else:
                        st.warning("No universities selected!")

            with col_stage2:
                if st.button("📧 Run Stage 2", help="Generate quality emails (higher cost)"):
                    if user_name:
                        with st.spinner("Running Stage 2: Email Generation..."):
                            try:
                                run_stage2_sync(
                                    st.session_state.research_profile, user_name, st.session_state.orchestrator)
                                st.success("✅ Stage 2 completed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Stage 2 failed: {e}")
                                logger.error(f"Stage 2 error: {e}")
                    else:
                        st.warning("Please enter your name in settings!")

            with col_stop:
                if st.button("⏹️ Stop", help="Stop current process"):
                    if st.session_state.orchestrator:
                        st.session_state.orchestrator.stop_research()
                        st.info("🛑 Process stopped")
        else:
            st.warning("⚠️ Please configure API key and upload CV first!")

    with col2:
        st.header("👨‍🏫 Research Results")

        if st.session_state.orchestrator:
            # Filter and display options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                status_filter = st.selectbox("Status Filter",
                                             ["All", "pending", "verified", "email_drafted", "email_sent"])
            with filter_col2:
                min_score = st.slider(
                    "Min Alignment Score", 0.0, 10.0, 6.0, 0.5)

            # Get professors from database
            try:
                conn = sqlite3.connect(
                    st.session_state.orchestrator.db.db_path)
                query = "SELECT * FROM professors WHERE alignment_score >= ?"
                params = [min_score]

                if status_filter != "All":
                    query += " AND status = ?"
                    params.append(status_filter)

                query += " ORDER BY alignment_score DESC"

                professors_df = pd.read_sql_query(query, conn, params=params)
                conn.close()

                if not professors_df.empty:
                    st.write(f"**{len(professors_df)} professors found**")

                    # Display professors
                    for idx, prof in professors_df.iterrows():
                        alignment_color = "🟢" if prof['alignment_score'] >= 8 else "🟡" if prof['alignment_score'] >= 6 else "🔴"

                        # Status badge
                        status_class = f"status-{prof['status'].replace('_', '-')}"
                        status_display = prof['status'].replace(
                            '_', ' ').title()

                        with st.expander(f"{alignment_color} {prof['name']} - {prof['university']} (Score: {prof['alignment_score']:.1f})"):
                            col_info, col_actions = st.columns([2, 1])

                            with col_info:
                                st.markdown(
                                    f'<span class="professor-status {status_class}">{status_display}</span>', unsafe_allow_html=True)
                                st.write(
                                    f"**Department:** {prof['department']}")
                                st.write(f"**Email:** {prof['email']}")
                                if prof['research_interests']:
                                    research_text = prof['research_interests'][:200] + (
                                        "..." if len(prof['research_interests']) > 200 else "")
                                    st.write(f"**Research:** {research_text}")
                                if prof['collaboration_potential']:
                                    st.write(
                                        f"**Collaboration Potential:** {prof['collaboration_potential']}")
                                if prof['profile_url']:
                                    st.write(
                                        f"**Profile:** [{prof['profile_url']}]({prof['profile_url']})")
                                st.write(
                                    f"**Costs:** Stage 1: ${prof['stage1_cost']:.4f}, Stage 2: ${prof['stage2_cost']:.4f}")

                                # Show email draft if available
                                if prof['draft_email_subject']:
                                    st.markdown("**📧 Draft Email:**")
                                    st.text_input("Subject:", value=prof['draft_email_subject'],
                                                  key=f"subject_{prof['id']}", disabled=True)
                                    email_body = prof['draft_email_body'][:300] + (
                                        "..." if len(prof['draft_email_body']) > 300 else "")
                                    st.text_area("Body:", value=email_body,
                                                 key=f"body_{prof['id']}", height=100, disabled=True)

                            with col_actions:
                                # Action buttons based on status
                                if prof['status'] == 'verified':
                                    if st.button("📧 Generate Email", key=f"gen_email_{prof['id']}"):
                                        if user_name:
                                            with st.spinner("Generating email..."):
                                                try:
                                                    success = generate_single_email_sync(
                                                        prof['id'], user_name, st.session_state.orchestrator,
                                                        st.session_state.research_profile
                                                    )
                                                    if success:
                                                        st.success(
                                                            "✅ Email generated!")
                                                        st.rerun()
                                                    else:
                                                        st.error(
                                                            "❌ Failed to generate email")
                                                except Exception as e:
                                                    st.error(f"❌ Error: {e}")
                                        else:
                                            st.warning(
                                                "Please enter your name in settings!")

                                elif prof['status'] == 'email_drafted':
                                    if st.button("👀 Preview Email", key=f"preview_{prof['id']}"):
                                        show_email_preview(prof)

                                    if st.button("📤 Send Email", key=f"send_{prof['id']}", type="primary"):
                                        st.info(
                                            "📧 Gmail integration: Email ready to send!")
                                        # Gmail sending would be implemented here

                                elif prof['status'] == 'email_sent':
                                    st.success("✅ Email Sent")
                                    if st.button("📬 View in Gmail", key=f"gmail_{prof['id']}"):
                                        st.info(
                                            "📬 Gmail integration: Open sent email")

                                # Delete button
                                if st.button("🗑️ Delete", key=f"delete_{prof['id']}"):
                                    try:
                                        conn = sqlite3.connect(
                                            st.session_state.orchestrator.db.db_path)
                                        cursor = conn.cursor()
                                        cursor.execute(
                                            "DELETE FROM professors WHERE id = ?", (prof['id'],))
                                        conn.commit()
                                        conn.close()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(
                                            f"Error deleting professor: {e}")
                else:
                    st.info(
                        "No professors found yet. Run Stage 1 to discover professors!")

            except Exception as e:
                st.error(f"Error loading professors: {e}")

    # Progress and status area
    st.markdown("---")
    st.header("📊 Live Progress & Analytics")

    if st.session_state.orchestrator:
        # Progress metrics
        prog_col1, prog_col2, prog_col3, prog_col4 = st.columns(4)

        try:
            conn = sqlite3.connect(st.session_state.orchestrator.db.db_path)

            # Get counts by status
            status_counts = pd.read_sql_query('''
                SELECT status, COUNT(*) as count, AVG(alignment_score) as avg_score,
                       SUM(stage1_cost) as total_stage1, SUM(stage2_cost) as total_stage2
                FROM professors 
                GROUP BY status
            ''', conn)

            conn.close()

            with prog_col1:
                pending = status_counts[status_counts['status'] == 'pending']['count'].sum(
                ) if not status_counts.empty else 0
                st.metric("⏳ Pending", int(pending))

            with prog_col2:
                verified = status_counts[status_counts['status'] == 'verified']['count'].sum(
                ) if not status_counts.empty else 0
                st.metric("✅ Verified", int(verified))

            with prog_col3:
                drafted = status_counts[status_counts['status'] == 'email_drafted']['count'].sum(
                ) if not status_counts.empty else 0
                st.metric("📧 Emails Drafted", int(drafted))

            with prog_col4:
                sent = status_counts[status_counts['status'] == 'email_sent']['count'].sum(
                ) if not status_counts.empty else 0
                st.metric("📤 Emails Sent", int(sent))

            # Cost breakdown chart
            if not status_counts.empty:
                st.subheader("💰 Cost Breakdown by Stage")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    total_stage1 = status_counts['total_stage1'].sum()
                    total_stage2 = status_counts['total_stage2'].sum()

                    cost_data = pd.DataFrame({
                        'Stage': ['Stage 1 (Discovery)', 'Stage 2 (Email Gen)'],
                        'Cost': [total_stage1, total_stage2],
                        'Description': ['gpt-4o-mini - Cost Effective', 'gpt-4 - High Quality']
                    })

                    if total_stage1 > 0 or total_stage2 > 0:
                        st.bar_chart(cost_data.set_index('Stage')['Cost'])

                with chart_col2:
                    st.dataframe(cost_data)

        except Exception as e:
            st.error(f"Error loading analytics: {e}")

        # Progress messages
        if st.session_state.orchestrator and hasattr(st.session_state.orchestrator, 'progress_messages'):
            st.subheader("🔄 Live Progress")

            # Show recent progress messages
            recent_messages = st.session_state.orchestrator.progress_messages[-10:]
            for message in recent_messages:
                st.markdown(
                    f'<div class="progress-message">{message}</div>', unsafe_allow_html=True)


def show_email_preview(professor_data):
    """Show email preview modal."""
    st.markdown("---")
    st.subheader(f"📧 Email Preview: {professor_data['name']}")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.text_input(
            "Subject:", value=professor_data['draft_email_subject'], key="preview_subject")
        st.text_area(
            "Email Body:", value=professor_data['draft_email_body'], height=300, key="preview_body")

    with col2:
        st.write("**Professor Info:**")
        st.write(f"Name: {professor_data['name']}")
        st.write(f"University: {professor_data['university']}")
        st.write(f"Score: {professor_data['alignment_score']:.1f}/10")

        if st.button("📤 Send Email", type="primary"):
            st.info(
                "📧 Email sending feature will be implemented with Gmail API integration")

        if st.button("✏️ Edit Email"):
            st.info("Email editing feature coming soon")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Add custom styling for Windows 11
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    
    .cost-efficient {
        background: linear-gradient(90deg, #e8f5e8, #d4edda);
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    
    .high-quality {
        background: linear-gradient(90deg, #e7f3ff, #d1ecf1);
        border-left: 4px solid #007bff;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    main()
