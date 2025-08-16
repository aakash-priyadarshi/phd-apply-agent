#!/usr/bin/env python3
"""
PhD Outreach Automation - Optimized 2-Stage System (Windows 11)
Stage 1: Extract & Match (gpt-4o-mini) - Cost-effective professor discovery
Stage 2: Email Drafting (gpt-4) - High-quality personalized emails
"""

import streamlit as st
import pandas as pd
import asyncio
import aiohttp
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

# no web driver

import requests
from bs4 import BeautifulSoup
import re

#

# Third-party imports
import openai
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import requests
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
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phd_outreach.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    # pending, researching, stage1_complete, stage2_complete, completed
    status: str = "pending"
    professors_found: int = 0
    professors_processed: int = 0
    total_stage1_cost: float = 0.0
    total_stage2_cost: float = 0.0
    last_scraped: str = ""


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


class OptimizedWebScraper:
    """Optimized web scraper for Windows 11 with cost-conscious extraction."""

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """Setup Chrome WebDriver with fixed driver path."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--timeout=20000")
        chrome_options.add_argument("--page-load-strategy=eager")

        try:
            # Use the fixed ChromeDriver path
            fixed_driver_path = os.path.join(
                os.getcwd(), "chromedriver_fixed", "chromedriver.exe")

            if os.path.exists(fixed_driver_path):
                service = Service(fixed_driver_path)
                logger.info(f"Using fixed ChromeDriver: {fixed_driver_path}")
            else:
                # Fallback to webdriver-manager
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                logger.info("Using webdriver-manager fallback")

            self.driver = webdriver.Chrome(
                service=service, options=chrome_options)
            self.driver.set_page_load_timeout(20)
            self.driver.implicitly_wait(5)

            logger.info("Chrome WebDriver initialized successfully")

        except Exception as e:
            logger.error(f"Chrome driver setup failed: {e}")
            self.driver = None

    def scrape_university_faculty_sync(self, university: str, departments: List[str],
                                       user_research_profile: str) -> List[Professor]:
        """Scrape university faculty using 2-stage approach - Sync version."""
        professors = []

        if not self.driver:
            logger.error("WebDriver not available")
            return professors

        try:
            for dept in departments:
                # Find faculty pages
                faculty_pages = self.find_faculty_pages_sync(university, dept)

                # Limit to top 2 pages per department to control costs
                for page_url in faculty_pages[:2]:
                    try:
                        # Get page content
                        self.driver.get(page_url)
                        time.sleep(3)

                        page_content = self.driver.page_source

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
                                    stage1_cost=stage1_result["cost"] /
                                    len(stage1_result["professors"]
                                        ) if stage1_result["professors"] else 0
                                )
                                professors.append(professor)

                        # Rate limiting
                        time.sleep(2)

                    except Exception as e:
                        logger.error(f"Error scraping {page_url}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error scraping {university}: {e}")

        return professors

    def find_faculty_pages_sync(self, university: str, department: str) -> List[str]:
        """Find faculty pages for a university department - Sync version."""
        faculty_urls = []

        try:
            # Search for faculty pages
            search_terms = [
                f"{university} {department} faculty",
                f"{university} {department} professors",
                f"{university} {department} people"
            ]

            for search_term in search_terms:
                try:
                    # Use Google search to find faculty pages
                    search_url = f"https://www.google.com/search?q={search_term.replace(' ', '+')}"

                    self.driver.get(search_url)
                    time.sleep(2)

                    # Extract search result links
                    links = self.driver.find_elements(
                        By.CSS_SELECTOR, "a[href]")

                    for link in links[:5]:  # Top 5 results
                        try:
                            href = link.get_attribute("href")
                            if href and any(term in href.lower() for term in ["faculty", "people", "staff"]):
                                if university.lower().replace(" ", "") in href.lower():
                                    faculty_urls.append(href)
                        except:
                            continue

                    if faculty_urls:
                        break  # Found pages, no need to continue searching

                except Exception as e:
                    logger.debug(f"Search error for {search_term}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error finding faculty pages: {e}")

        return list(set(faculty_urls))  # Remove duplicates

    def close_driver(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()


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
                    prof_id = self.db.add_professor(professor)
                    professor_ids.append(prof_id)

                # Update cost tracking
                stage1_cost = sum(p.stage1_cost for p in professors)
                self.db.update_cost_tracking(
                    stage1_cost, 0.0, len(professors), 0)

                self.add_progress_message(
                    f"✅ Stage 1 Complete: {university.name} - {len(professors)} professors found "
                    f"(Cost: ${stage1_cost:.4f})"
                )

                time.sleep(3)  # Rate limiting between universities

        except Exception as e:
            logger.error(f"Stage 1 error: {e}")
            self.add_progress_message(f"❌ Stage 1 error: {e}")
        finally:
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
            self.db.update_cost_tracking(
                0.0, total_stage2_cost, 0, emails_generated)

            self.add_progress_message(
                f"✅ Stage 2 Complete: {emails_generated} emails generated "
                f"(Total cost: ${total_stage2_cost:.4f})"
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
            self.web_scraper.close_driver()


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
    }
    .progress-message {
        background: #f8f9fa;
        padding: 0.3rem;
        border-radius: 3px;
        margin: 0.2rem 0;
        font-family: monospace;
        font-size: 0.8rem;
    }
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
                                st.experimental_rerun()

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
                        st.experimental_rerun()
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
                '<div class="stage-indicator"><strong>Stage 1:</strong> Professor Discovery & Matching (gpt-4o-mini - Cost Effective)</div>',
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
                                st.experimental_rerun()
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
                                st.experimental_rerun()
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

                        with st.expander(f"{alignment_color} {prof['name']} - {prof['university']} (Score: {prof['alignment_score']:.1f})"):
                            col_info, col_actions = st.columns([2, 1])

                            with col_info:
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
                                st.write(f"**Status:** {prof['status']}")
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
                                                        st.experimental_rerun()
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
                                        st.experimental_rerun()
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
