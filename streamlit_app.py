"""
PhD Outreach Automation - Complete 2-Stage System
Stage 1: Extract & Match (gpt-4o-mini) - Cost-effective professor discovery
Stage 2: Email Drafting (gpt-4) - High-quality personalized emails
"""

# Standard library imports
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
import re

# Web scraping imports
import requests
from bs4 import BeautifulSoup

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

# Import Gmail manager
try:
    from gmail_manager import GmailManager, EmailTemplateManager, create_bulk_email_data
    GMAIL_AVAILABLE = True
except ImportError:
    print("Gmail manager not available - email sending will be disabled")
    GmailManager = None
    GMAIL_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phd_outreach.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_log_message(message: str) -> str:
    """Remove emojis from log messages to prevent encoding errors."""
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
    alignment_score: float = 0.0
    collaboration_potential: str = ""
    last_verified: str = ""
    status: str = "pending"
    created_at: str = ""
    email_sent_at: str = ""
    draft_email_subject: str = ""
    draft_email_body: str = ""
    notes: str = ""
    stage1_cost: float = 0.0
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

        # Pricing per 1K tokens
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
        """Stage 1: Extract professor info and match with user research (gpt-4o-mini)."""
        prompt = f"""
        Extract professor information from this faculty page content and match with user's research profile.
        
        Faculty Page Content:
        {page_content[:4000]}
        
        User's Research Profile:
        {user_research_profile}
        
        Extract up to 10 professors and for each provide:
        1. Name and basic info
        2. Research areas/interests
        3. Alignment score (1-10) with user's profile
        4. Brief reasoning for the score
        
        IMPORTANT: Use only standard ASCII characters in the JSON response. No tabs, newlines, or special characters within strings.
        
        Output as clean JSON array:
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

            cost = self.calculate_cost(
                "gpt-4o-mini",
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0
            )
            self.stage1_cost += cost
            self.total_cost += cost

            content = response.choices[0].message.content or ""

            # Clean the content to remove potential control characters
            cleaned_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)

            # Parse JSON response
            try:
                json_match = re.search(r'\[.*\]', cleaned_content, re.DOTALL)
                if json_match:
                    json_text = json_match.group()
                    # Additional cleaning for common JSON issues
                    json_text = json_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    
                    professors_data = json.loads(json_text)
                    return {
                        "professors": professors_data,
                        "cost": cost,
                        "success": True
                    }
                else:
                    return {"professors": [], "cost": cost, "success": False, "error": "No JSON array found"}
            except json.JSONDecodeError as e:
                logger.error(f"Stage 1 JSON parse error: {e}, Content: {cleaned_content[:200]}")
                return {"professors": [], "cost": cost, "success": False, "error": f"JSON parse error: {e}"}

        except Exception as e:
            logger.error(f"Stage 1 API error: {e}")
            return {"professors": [], "cost": 0, "success": False, "error": str(e)}

    def stage2_draft_email_sync(self, professor: Professor, user_research_profile: str, user_name: str) -> Dict[str, Any]:
        """Stage 2: Draft personalized email (gpt-4)."""
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
        1. Professional academic tone with proper email formatting
        2. Specific reference to professor's research
        3. Clear research alignment explanation
        4. Mention attached CV
        5. Request for PhD opportunity discussion
        6. Keep email body under 180 words
        7. Subject line under 80 characters
        8. Format the email body with proper paragraphs separated by "\\n\\n"
        9. Include greeting, 2-3 body paragraphs, closing, and signature
        10. IMPORTANT: Use only standard ASCII characters in the JSON response. No tabs, newlines, or special characters within strings.
        
        Email Structure:
        - Greeting: "Dear Dr. [Last Name],"
        - Opening paragraph: Brief introduction and purpose
        - Body paragraph(s): Research alignment and specific interests
        - Closing paragraph: Request for discussion and mention CV attachment
        - Professional closing: "Best regards," followed by applicant name
        
        Output as clean JSON with escaped strings:
        {{
            "subject": "Compelling subject line mentioning specific research area",
            "body": "Dear Dr. [Professor Last Name],\\n\\nOpening paragraph with introduction and purpose.\\n\\nBody paragraph with specific research alignment and interests.\\n\\nClosing paragraph with request for discussion and CV mention.\\n\\nBest regards,\\n{user_name}",
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

            cost = self.calculate_cost(
                "gpt-4",
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0
            )
            self.stage2_cost += cost
            self.total_cost += cost

            content = response.choices[0].message.content or ""

            # Clean the content to remove potential control characters
            cleaned_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)
            
            # Parse JSON response with better error handling
            try:
                json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
                if json_match:
                    json_text = json_match.group()
                    # Additional cleaning for common JSON issues
                    json_text = json_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    
                    email_data = json.loads(json_text)
                    
                    # Validate required fields
                    if not email_data.get("subject") or not email_data.get("body"):
                        return {"success": False, "cost": cost, "error": "Missing required fields in response"}
                    
                    email_data["cost"] = cost
                    email_data["success"] = True
                    return email_data
                else:
                    # Fallback: try to extract subject and body manually
                    return self._extract_email_fallback(cleaned_content, cost)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}, Content: {cleaned_content[:200]}")
                # Try fallback extraction
                return self._extract_email_fallback(cleaned_content, cost)

        except Exception as e:
            logger.error(f"Stage 2 API error: {e}")
            return {"success": False, "cost": 0, "error": str(e)}

    def _extract_email_fallback(self, content: str, cost: float) -> Dict[str, Any]:
        """Fallback method to extract email content when JSON parsing fails."""
        try:
            # Try to extract subject and body using regex patterns
            subject_match = re.search(r'["\']?subject["\']?\s*[:=]\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
            body_match = re.search(r'["\']?body["\']?\s*[:=]\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
            
            if subject_match and body_match:
                return {
                    "success": True,
                    "cost": cost,
                    "subject": subject_match.group(1).strip(),
                    "body": body_match.group(1).strip(),
                    "key_points": ["Extracted from fallback"],
                    "tone_analysis": "professional"
                }
            else:
                # Last resort: generate a basic template
                return {
                    "success": True,
                    "cost": cost,
                    "subject": f"PhD Application - Research Interest in {content[:50]}",
                    "body": f"Dear Professor,\n\nI am writing to express my interest in pursuing a PhD under your supervision. {content[:100]}...\n\nBest regards,\n{content}",
                    "key_points": ["Fallback template"],
                    "tone_analysis": "professional"
                }
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return {"success": False, "cost": cost, "error": f"Fallback extraction failed: {e}"}


class DatabaseManager:
    """Enhanced database manager with cost tracking."""

    def __init__(self, db_path: str = "phd_outreach.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables with cost tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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

        conn.commit()
        conn.close()

    def add_professor(self, professor: Professor) -> int:
        """Add professor with enhanced data and duplicate check."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check for duplicates based on name and university
        cursor.execute('''
            SELECT id FROM professors 
            WHERE name = ? AND university = ? AND email = ?
        ''', (professor.name, professor.university, professor.email))
        
        existing = cursor.fetchone()
        if existing:
            # Update existing professor instead of creating duplicate
            cursor.execute('''
                UPDATE professors SET
                    department = ?, research_interests = ?, ongoing_research = ?,
                    profile_url = ?, alignment_score = ?, collaboration_potential = ?,
                    last_verified = ?, status = ?, notes = ?, stage1_cost = stage1_cost + ?
                WHERE id = ?
            ''', (
                professor.department, professor.research_interests, professor.ongoing_research,
                professor.profile_url, professor.alignment_score, professor.collaboration_potential,
                professor.last_verified, professor.status, professor.notes, professor.stage1_cost,
                existing[0]
            ))
            conn.commit()
            conn.close()
            return existing[0]

        # Insert new professor
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
        return professor_id or 0

    def update_cost_tracking(self, stage1_cost: float, stage2_cost: float,
                             professors_processed: int, emails_generated: int):
        """Update daily cost tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute('SELECT id FROM cost_tracking WHERE date = ?', (today,))
        record = cursor.fetchone()

        if record:
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

    def clean_duplicate_professors(self):
        """Remove duplicate professors based on name and university."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find duplicates
        cursor.execute('''
            SELECT name, university, email, COUNT(*) as count,
                   GROUP_CONCAT(id) as ids
            FROM professors 
            GROUP BY name, university, email
            HAVING count > 1
        ''')
        
        duplicates = cursor.fetchall()
        removed_count = 0
        
        for name, university, email, count, ids in duplicates:
            id_list = ids.split(',')
            # Keep the first one (oldest), remove others
            ids_to_remove = id_list[1:]
            
            for prof_id in ids_to_remove:
                cursor.execute('DELETE FROM professors WHERE id = ?', (prof_id,))
                removed_count += 1
        
        conn.commit()
        conn.close()
        
        return removed_count

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute('SELECT * FROM cost_tracking WHERE date = ?', (today,))
        today_data = cursor.fetchone()

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
    """Pure HTTP scraper using requests + BeautifulSoup."""

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def scrape_university_faculty_sync(self, university: str, departments: List[str], user_research_profile: str) -> List[Professor]:
        """Scrape university faculty using pure HTTP requests."""
        professors = []
        logger.info(f"Starting scrape for {university}")

        try:
            for dept in departments:
                faculty_pages = self.find_faculty_pages_sync(university, dept)
                logger.info(f"Found {len(faculty_pages)} faculty pages for {dept}")

                for page_url in faculty_pages[:2]:
                    try:
                        response = self.session.get(page_url, timeout=20)
                        response.raise_for_status()

                        soup = BeautifulSoup(response.content, 'html.parser')
                        page_content = soup.get_text()

                        logger.info(f"Scraped content from {page_url} - {len(page_content)} chars")

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
                                    research_interests=prof_data.get("research_interests", ""),
                                    profile_url=prof_data.get("profile_url", ""),
                                    alignment_score=prof_data.get("alignment_score", 0.0),
                                    collaboration_potential=prof_data.get("collaboration_potential", ""),
                                    status="verified" if prof_data.get("alignment_score", 0) >= 6.0 else "pending",
                                    created_at=datetime.now().isoformat(),
                                    last_verified=datetime.now().isoformat(),
                                    stage1_cost=stage1_result["cost"] / len(stage1_result["professors"]) if stage1_result["professors"] else 0
                                )
                                professors.append(professor)

                            logger.info(f"Stage 1 success: {len(stage1_result['professors'])} professors extracted")
                        else:
                            logger.warning(f"Stage 1 failed: {stage1_result.get('error')}")

                        time.sleep(2)

                    except requests.RequestException as e:
                        logger.error(f"HTTP error scraping {page_url}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing {page_url}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error scraping {university}: {e}")

        logger.info(f"Completed scrape for {university}: {len(professors)} professors found")
        return professors

    def find_faculty_pages_sync(self, university: str, department: str) -> List[str]:
        """Find faculty pages using direct university URLs."""
        faculty_urls = []

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
            }
        }

        university_info = university_mappings.get(university, {})
        dept_lower = department.lower()
        dept_urls = university_info.get(f"{dept_lower}_urls", [])

        for url in dept_urls[:2]:
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

    def __init__(self, openai_api_key: str, gmail_credentials: str = "credentials.json"):
        self.api_manager = APIManager(openai_api_key)
        self.web_scraper = NoWebDriverScraper(self.api_manager)
        self.db = DatabaseManager()

        # Gmail setup
        self.gmail_manager = None
        if GMAIL_AVAILABLE and GmailManager and os.path.exists(gmail_credentials):
            try:
                self.gmail_manager = GmailManager(gmail_credentials)
                logger.info("Gmail manager initialized successfully")
            except Exception as e:
                logger.warning(f"Gmail manager initialization failed: {e}")
                self.gmail_manager = None
        else:
            logger.warning("Gmail credentials not found or GmailManager not available")

        self.is_running = False
        self.progress_messages = []

    def add_progress_message(self, message: str):
        """Add progress message for UI."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.progress_messages.append(formatted_message)
        if len(self.progress_messages) > 20:
            self.progress_messages = self.progress_messages[-20:]
        logger.info(clean_log_message(message))

    def run_stage1_research_sync(self, universities: List[University], user_research_profile: str):
        """Run Stage 1: Extract and match professors."""
        self.is_running = True
        self.add_progress_message("🚀 Starting Stage 1: Professor Discovery & Matching...")

        try:
            for i, university in enumerate(universities):
                if not self.is_running:
                    break

                self.add_progress_message(f"🔍 Stage 1: Researching {university.name} ({i+1}/{len(universities)})...")

                departments = [dept.strip() for dept in university.departments.split(',')]
                professors = self.web_scraper.scrape_university_faculty_sync(
                    university.name, departments, user_research_profile
                )

                professor_ids = []
                for professor in professors:
                    prof_id = safe_operation(self.db.add_professor, professor)
                    if prof_id:
                        professor_ids.append(prof_id)

                stage1_cost = sum(p.stage1_cost for p in professors)
                safe_operation(self.db.update_cost_tracking, stage1_cost, 0.0, len(professors), 0)

                self.add_progress_message(f"✅ Stage 1 Complete: {university.name} - {len(professors)} professors found (Cost: ${stage1_cost:.4f})")

                time.sleep(3)

        except Exception as e:
            logger.error(f"Stage 1 error: {e}")
            self.add_progress_message(f"❌ Stage 1 error: {e}")
        finally:
            self.is_running = False
            self.add_progress_message("✅ Stage 1 Complete: Professor discovery finished!")

    def run_stage2_email_generation_sync(self, user_research_profile: str, user_name: str):
        """Run Stage 2: Generate high-quality personalized emails."""
        self.add_progress_message("📧 Starting Stage 2: Email Generation...")

        try:
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
                self.add_progress_message("ℹ️ No professors need email generation")
                return

            total_professors = len(professor_rows)
            self.add_progress_message(f"📧 Generating emails for {total_professors} professors...")

            emails_generated = 0
            total_stage2_cost = 0.0

            for i, row in enumerate(professor_rows):
                if not self.is_running:
                    break

                professor_data = dict(zip(columns, row))
                professor = Professor(**{k: v for k, v in professor_data.items() if k != 'id'})

                self.add_progress_message(f"✍️ Generating email for {professor.name} (Score: {professor.alignment_score:.1f}) [{i+1}/{total_professors}]")

                email_result = self.api_manager.stage2_draft_email_sync(professor, user_research_profile, user_name)

                if email_result["success"]:
                    conn = sqlite3.connect(self.db.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE professors 
                        SET draft_email_subject = ?, draft_email_body = ?, 
                            stage2_cost = ?, status = 'email_drafted'
                        WHERE id = ?
                    ''', (email_result["subject"], email_result["body"], email_result["cost"], professor_data['id']))
                    conn.commit()
                    conn.close()

                    emails_generated += 1
                    total_stage2_cost += email_result["cost"]

                    self.add_progress_message(f"✅ Email generated for {professor.name} (Cost: ${email_result['cost']:.4f})")
                else:
                    self.add_progress_message(f"❌ Failed to generate email for {professor.name}")

                time.sleep(1)

            safe_operation(self.db.update_cost_tracking, 0.0, total_stage2_cost, 0, emails_generated)

            self.add_progress_message(f"✅ Stage 2 Complete: {emails_generated} emails generated (Total cost: ${total_stage2_cost:.4f})")

        except Exception as e:
            logger.error(f"Stage 2 error: {e}")
            self.add_progress_message(f"❌ Stage 2 error: {e}")

    def generate_bulk_emails_sync(self, user_research_profile: str, user_name: str, status_filter: str = "verified") -> Dict[str, Any]:
        """Generate emails for all professors matching criteria."""
        self.add_progress_message("📧 Starting bulk email generation...")
        self.is_running = True
        
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            if status_filter == "verified":
                cursor.execute('''
                    SELECT * FROM professors 
                    WHERE status = 'verified' AND alignment_score >= 6.0 
                    AND (draft_email_subject IS NULL OR draft_email_subject = '')
                    ORDER BY alignment_score DESC
                ''')
            else:
                cursor.execute('''
                    SELECT * FROM professors 
                    WHERE status = ? AND alignment_score >= 6.0 
                    ORDER BY alignment_score DESC
                ''', (status_filter,))

            columns = [description[0] for description in cursor.description]
            professor_rows = cursor.fetchall()
            conn.close()

            if not professor_rows:
                self.add_progress_message("ℹ️ No professors need email generation")
                return {"success": True, "message": "No professors need email generation", "emails_generated": 0, "total_cost": 0.0}

            total_professors = len(professor_rows)
            self.add_progress_message(f"📧 Found {total_professors} professors for email generation...")

            emails_generated = 0
            total_cost = 0.0
            failed_generations = []

            for i, row in enumerate(professor_rows):
                if not self.is_running:
                    self.add_progress_message("⏹️ Process stopped by user")
                    break

                professor_data = dict(zip(columns, row))
                professor = Professor(**{k: v for k, v in professor_data.items() if k != 'id'})

                self.add_progress_message(f"✍️ [{i+1}/{total_professors}] Generating email for {professor.name} (Score: {professor.alignment_score:.1f})")

                # Add a small delay to prevent rate limiting
                if i > 0:
                    time.sleep(2)

                email_result = self.api_manager.stage2_draft_email_sync(professor, user_research_profile, user_name)

                if email_result.get("success", False):
                    try:
                        conn = sqlite3.connect(self.db.db_path)
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE professors 
                            SET draft_email_subject = ?, draft_email_body = ?, 
                                stage2_cost = ?, status = 'email_drafted'
                            WHERE id = ?
                        ''', (email_result["subject"], email_result["body"], email_result["cost"], professor_data['id']))
                        conn.commit()
                        conn.close()

                        emails_generated += 1
                        total_cost += email_result["cost"]

                        self.add_progress_message(f"✅ Email generated for {professor.name} (Cost: ${email_result['cost']:.4f})")
                    except Exception as db_error:
                        failed_generations.append({"name": professor.name, "error": f"Database error: {db_error}"})
                        self.add_progress_message(f"❌ Database error for {professor.name}: {db_error}")
                else:
                    error_msg = email_result.get("error", "Unknown error")
                    failed_generations.append({"name": professor.name, "error": error_msg})
                    self.add_progress_message(f"❌ Failed: {professor.name} - {error_msg}")

            # Update cost tracking
            safe_operation(self.db.update_cost_tracking, 0.0, total_cost, 0, emails_generated)

            success_message = f"✅ Bulk generation complete: {emails_generated}/{total_professors} emails generated (Cost: ${total_cost:.4f})"
            if failed_generations:
                success_message += f" | {len(failed_generations)} failed"
            
            self.add_progress_message(success_message)

            return {
                "success": True,
                "emails_generated": emails_generated,
                "total_professors": total_professors,
                "total_cost": total_cost,
                "failed_generations": failed_generations
            }

        except Exception as e:
            error_msg = f"Bulk email generation error: {e}"
            logger.error(error_msg)
            self.add_progress_message(f"❌ {error_msg}")
            return {"success": False, "error": str(e)}
        finally:
            self.is_running = False
        
###################################################################

    def send_bulk_emails_sync(self, user_name: str, cv_path: str = "uploaded_cv.pdf", delay_seconds: int = 10) -> Dict[str, Any]:
            """Send all drafted emails with Gmail API."""
            
            if not self.gmail_manager:
                return {"success": False, "error": "Gmail not configured"}

            self.add_progress_message("📤 Starting bulk email sending...")
            self.is_running = True

            try:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM professors 
                    WHERE status = 'email_drafted' 
                    AND draft_email_subject IS NOT NULL 
                    AND draft_email_body IS NOT NULL
                    AND email != ''
                    ORDER BY alignment_score DESC
                ''')

                columns = [description[0] for description in cursor.description]
                professor_rows = cursor.fetchall()
                conn.close()

                if not professor_rows:
                    return {"success": True, "message": "No emails ready to send", "emails_sent": 0}

                total_emails = len(professor_rows)
                self.add_progress_message(f"📤 Sending {total_emails} emails...")

                email_data_list = []
                for row in professor_rows:
                    professor_data = dict(zip(columns, row))
                    
                    if not self.gmail_manager.validate_email_address(professor_data['email']):
                        self.add_progress_message(f"❌ Invalid email for {professor_data['name']}: {professor_data['email']}")
                        continue
                    
                    email_data_list.append({
                        'professor_id': professor_data['id'],
                        'professor_name': professor_data['name'],
                        'university': professor_data['university'],
                        'to_email': professor_data['email'],
                        'subject': professor_data['draft_email_subject'],
                        'body': professor_data['draft_email_body']
                    })

                if not email_data_list:
                    return {"success": False, "error": "No valid emails to send"}

                emails_sent = 0
                failed_sends = []

                for i, email_data in enumerate(email_data_list):
                    if not self.is_running:
                        break

                    self.add_progress_message(f"📧 [{i+1}/{len(email_data_list)}] Sending to {email_data['professor_name']} ({email_data['to_email']})")

                    send_result = self.gmail_manager.send_email(
                        to_email=email_data['to_email'],
                        subject=email_data['subject'],
                        body=email_data['body'],
                        from_name=user_name,
                        cv_path=cv_path if cv_path and os.path.exists(cv_path) else ""
                    )

                    if send_result["success"]:
                        conn = sqlite3.connect(self.db.db_path)
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE professors 
                            SET status = 'email_sent', email_sent_at = ?
                            WHERE id = ?
                        ''', (send_result['sent_at'], email_data['professor_id']))
                        conn.commit()
                        conn.close()

                        emails_sent += 1
                        self.add_progress_message(f"✅ Email sent to {email_data['professor_name']} (ID: {send_result['message_id'][:10]}...)")
                    else:
                        failed_sends.append({
                            "professor_name": email_data['professor_name'],
                            "email": email_data['to_email'],
                            "error": send_result.get('error', 'Unknown error')
                        })
                        self.add_progress_message(f"❌ Failed to send to {email_data['professor_name']}: {send_result.get('error', 'Unknown error')}")

                    if i < len(email_data_list) - 1 and delay_seconds > 0:
                        self.add_progress_message(f"⏳ Waiting {delay_seconds} seconds...")
                        time.sleep(delay_seconds)

                self.add_progress_message(f"✅ Bulk sending complete: {emails_sent}/{len(email_data_list)} emails sent")

                return {
                    "success": True,
                    "emails_sent": emails_sent,
                    "total_emails": len(email_data_list),
                    "failed_sends": failed_sends
                }

            except Exception as e:
                logger.error(f"Bulk email sending error: {e}")
                self.add_progress_message(f"❌ Bulk sending error: {e}")
                return {"success": False, "error": str(e)}
            finally:
                self.is_running = False

    def generate_and_send_all_sync(self, user_research_profile: str, user_name: str, cv_path: str = "uploaded_cv.pdf", delay_seconds: int = 10) -> Dict[str, Any]:
        """Generate emails for all verified professors and send them immediately."""
        self.add_progress_message("🚀 Starting generate and send all process...")

        try:
            generation_result = self.generate_bulk_emails_sync(user_research_profile, user_name)
            
            if not generation_result["success"]:
                return generation_result

            if generation_result["emails_generated"] == 0:
                return {
                    "success": True, 
                    "message": "No emails to generate and send",
                    "emails_generated": 0,
                    "emails_sent": 0
                }

            self.add_progress_message(f"✅ Generated {generation_result['emails_generated']} emails, now sending...")

            time.sleep(2)

            sending_result = self.send_bulk_emails_sync(user_name, cv_path, delay_seconds)

            if sending_result["success"]:
                total_cost = generation_result.get("total_cost", 0.0)
                
                self.add_progress_message(f"🎉 Complete! Generated {generation_result['emails_generated']} emails, sent {sending_result['emails_sent']} (Cost: ${total_cost:.4f})")

                return {
                    "success": True,
                    "emails_generated": generation_result["emails_generated"],
                    "emails_sent": sending_result["emails_sent"],
                    "total_cost": total_cost,
                    "failed_generations": generation_result.get("failed_generations", []),
                    "failed_sends": sending_result.get("failed_sends", [])
                }
            else:
                return sending_result

        except Exception as e:
            logger.error(f"Generate and send all error: {e}")
            self.add_progress_message(f"❌ Generate and send all error: {e}")
            return {"success": False, "error": str(e)}

    def generate_single_email_sync(self, professor_id: int, user_name: str, user_research_profile: str) -> bool:
        """Generate email for a single professor."""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM professors WHERE id = ?', (professor_id,))

            columns = [description[0] for description in cursor.description]
            row = cursor.fetchone()
            conn.close()

            if row:
                professor_data = dict(zip(columns, row))
                professor = Professor(**{k: v for k, v in professor_data.items() if k != 'id'})

                email_result = self.api_manager.stage2_draft_email_sync(professor, user_research_profile, user_name)

                if email_result["success"]:
                    conn = sqlite3.connect(self.db.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE professors 
                        SET draft_email_subject = ?, draft_email_body = ?, 
                            stage2_cost = ?, status = 'email_drafted'
                        WHERE id = ?
                    ''', (email_result["subject"], email_result["body"], email_result["cost"], professor_id))
                    conn.commit()
                    conn.close()

                    self.add_progress_message(f"✅ Email generated for {professor.name}")
                    return True
                else:
                    self.add_progress_message(f"❌ Failed to generate email: {email_result.get('error', 'Unknown error')}")
                    return False
            else:
                self.add_progress_message("❌ Professor not found")
                return False

        except Exception as e:
            logger.error(f"Error generating email for professor {professor_id}: {e}")
            self.add_progress_message(f"❌ Error generating email: {e}")
            return False

    def send_single_email_sync(self, professor_id: int, user_name: str, cv_path: str = "uploaded_cv.pdf") -> bool:
        """Send email for a single professor."""
        
        if not self.gmail_manager:
            self.add_progress_message("❌ Gmail not configured")
            return False

        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM professors WHERE id = ?', (professor_id,))
            
            columns = [description[0] for description in cursor.description]
            row = cursor.fetchone()
            conn.close()

            if not row:
                self.add_progress_message("❌ Professor not found")
                return False

            professor_data = dict(zip(columns, row))
            
            if professor_data['status'] != 'email_drafted':
                self.add_progress_message("❌ Email not drafted for this professor")
                return False

            if not professor_data['email']:
                self.add_progress_message("❌ No email address for this professor")
                return False

            send_result = self.gmail_manager.send_email(
                to_email=professor_data['email'],
                subject=professor_data['draft_email_subject'],
                body=professor_data['draft_email_body'],
                from_name=user_name,
                cv_path=cv_path if cv_path and os.path.exists(cv_path) else ""
            )

            if send_result["success"]:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE professors 
                    SET status = 'email_sent', email_sent_at = ?
                    WHERE id = ?
                ''', (send_result['sent_at'], professor_id))
                conn.commit()
                conn.close()

                self.add_progress_message(f"✅ Email sent to {professor_data['name']} ({professor_data['email']})")
                return True
            else:
                self.add_progress_message(f"❌ Failed to send email to {professor_data['name']}: {send_result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Error sending email for professor {professor_id}: {e}")
            self.add_progress_message(f"❌ Error sending email: {e}")
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


def send_single_email_sync(professor_id: int, user_name: str, orchestrator: ResearchOrchestrator) -> bool:
    """Synchronous wrapper for single email sending."""
    return orchestrator.send_single_email_sync(professor_id, user_name)


def generate_bulk_emails_sync(user_research_profile: str, user_name: str, orchestrator: ResearchOrchestrator) -> Dict[str, Any]:
    """Synchronous wrapper for bulk email generation."""
    return orchestrator.generate_bulk_emails_sync(user_research_profile, user_name)


def send_bulk_emails_sync(user_name: str, orchestrator: ResearchOrchestrator, delay_seconds: int = 10) -> Dict[str, Any]:
    """Synchronous wrapper for bulk email sending."""
    return orchestrator.send_bulk_emails_sync(user_name, delay_seconds=delay_seconds)


def generate_and_send_all_sync(user_research_profile: str, user_name: str, orchestrator: ResearchOrchestrator, delay_seconds: int = 10) -> Dict[str, Any]:
    """Synchronous wrapper for generate and send all."""
    return orchestrator.generate_and_send_all_sync(user_research_profile, user_name, delay_seconds=delay_seconds)


@st.dialog("Edit Email")
def show_email_edit_modal(professor_data):
    """Show email editing in a modal dialog."""
    st.subheader(f"✏️ Edit Email for {professor_data['name']}")
    
    # Professor info at the top
    with st.expander("�‍🏫 Professor Information", expanded=False):
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write(f"**Name:** {professor_data['name']}")
            st.write(f"**University:** {professor_data['university']}")
        with info_col2:
            st.write(f"**Department:** {professor_data.get('department', 'N/A')}")
            st.write(f"**Email:** {professor_data.get('email', 'N/A')}")
        
        if professor_data.get('research_interests'):
            st.write(f"**Research:** {professor_data['research_interests']}")
        if professor_data.get('alignment_score'):
            st.write(f"**Alignment Score:** {professor_data['alignment_score']:.1f}/10")

    # Editable email fields
    st.markdown("### 📧 Email Content")
    
    edited_subject = st.text_input(
        "Subject Line:", 
        value=professor_data.get('draft_email_subject', ''), 
        key=f"modal_subject_{professor_data['id']}",
        help="Keep it under 80 characters for better deliverability")
    
    # Character count for subject
    subject_length = len(edited_subject or "")
    if subject_length > 80:
        st.warning(f"Subject is {subject_length} characters (recommended: ≤80)")
    else:
        st.info(f"Subject length: {subject_length}/80 characters")
    
    edited_body = st.text_area(
        "Email Body:", 
        value=professor_data.get('draft_email_body', '').replace('\\n\\n', '\n\n').replace('\\n', '\n'), 
        height=300, 
        key=f"modal_body_{professor_data['id']}",
        help="Keep it professional and under 180 words")
    
    # Word count for body
    word_count = len((edited_body or "").split())
    if word_count > 180:
        st.warning(f"Email is {word_count} words (recommended: ≤180)")
    else:
        st.info(f"Word count: {word_count}/180 words")

    # Preview section
    with st.expander("👀 Email Preview", expanded=False):
        st.markdown("**Subject:**")
        st.code(edited_subject or "")
        st.markdown("**Body:**")
        st.markdown(edited_body or "")

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Changes", key=f"modal_save_{professor_data['id']}", type="primary"):
            try:
                # Convert line breaks back to escaped format for storage
                saved_body = (edited_body or "").replace('\n\n', '\\n\\n').replace('\n', '\\n')
                
                conn = sqlite3.connect("phd_outreach.db")
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE professors 
                    SET draft_email_subject = ?, draft_email_body = ?
                    WHERE id = ?
                ''', (edited_subject, saved_body, professor_data['id']))
                conn.commit()
                conn.close()
                
                st.success("✅ Email changes saved!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save changes: {e}")
    
    with col2:
        if st.button("📤 Save & Send", key=f"modal_send_{professor_data['id']}"):
            try:
                # Convert line breaks back to escaped format for storage
                saved_body = (edited_body or "").replace('\n\n', '\\n\\n').replace('\n', '\\n')
                
                # First save the changes
                conn = sqlite3.connect("phd_outreach.db")
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE professors 
                    SET draft_email_subject = ?, draft_email_body = ?
                    WHERE id = ?
                ''', (edited_subject, saved_body, professor_data['id']))
                conn.commit()
                conn.close()
                
                # Then send the email
                if st.session_state.orchestrator and st.session_state.orchestrator.gmail_manager:
                    user_name = os.getenv('USER_NAME', 'PhD Applicant')
                    success = send_single_email_sync(professor_data['id'], user_name, st.session_state.orchestrator)
                    if success:
                        st.success("✅ Email sent successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to send email")
                else:
                    st.error("❌ Gmail not configured")
                
            except Exception as e:
                st.error(f"❌ Failed to send email: {e}")

    with col3:
        if st.button("🔄 Regenerate", key=f"modal_regen_{professor_data['id']}"):
            try:
                user_name = os.getenv('USER_NAME', 'PhD Applicant')
                if user_name and st.session_state.research_profile:
                    with st.spinner("Regenerating email..."):
                        success = generate_single_email_sync(professor_data['id'], user_name, st.session_state.orchestrator, st.session_state.research_profile)
                        if success:
                            st.success("✅ Email regenerated!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Failed to regenerate email")
                else:
                    st.error("❌ Missing user name or research profile")
            except Exception as e:
                st.error(f"❌ Error regenerating email: {e}")

    with col4:
        if st.button("❌ Close", key=f"modal_close_{professor_data['id']}"):
            st.rerun()


def show_email_preview(professor_data):
    """Show comprehensive email preview with editing capabilities.""" 
    # This function now just opens the modal
    show_email_edit_modal(professor_data)


def main():
    """Main Streamlit application function."""
    st.set_page_config(
        page_title="PhD Outreach Automation - 2-Stage System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS styling
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
    .status-email-sent { background: #d1f2eb; color: #0c4128; }
    
    /* Enhanced modal and button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Professor card improvements */
    .professor-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .professor-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Alignment score styling */
    .alignment-score {
        font-weight: 700;
        font-size: 1.1em;
    }
    .score-high { color: #16a34a; }
    .score-medium { color: #d97706; }
    .score-low { color: #dc2626; }
    
    /* Modal improvements */
    [data-testid="stModal"] {
        width: 80vw !important;
        max-width: 900px !important;
    }
    
    [data-testid="stModal"] .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Email preview styling */
    .email-preview {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 8px 0;
    }
    
    /* Gmail status */
    .gmail-status-connected { color: #28a745; font-weight: bold; }
    .gmail-status-disconnected { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header"><h1>🎓 PhD Outreach Automation - 2-Stage System</h1><p>Stage 1: Smart Discovery (gpt-4o-mini) • Stage 2: Quality Emails (gpt-4)</p></div>', unsafe_allow_html=True)

    # Initialize session state
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'research_profile' not in st.session_state:
        st.session_state.research_profile = ""
        # Try to load existing research profile on app start
        research_profile_file = "research_profile.txt"
        if os.path.exists(research_profile_file):
            try:
                with open(research_profile_file, 'r', encoding='utf-8') as f:
                    saved_profile = f.read().strip()
                    if saved_profile:
                        st.session_state.research_profile = saved_profile
                        st.session_state.cv_analyzed = True
            except Exception:
                pass  # Silently fail, user can re-analyze if needed
    if 'cv_uploaded' not in st.session_state:
        st.session_state.cv_uploaded = False

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Keys
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv('OPENAI_API_KEY', ''))

        if openai_key:
            if st.session_state.orchestrator is None:
                try:
                    st.session_state.orchestrator = ResearchOrchestrator(openai_key, "credentials.json")
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
                    st.metric("Today's Cost", f"${cost_summary['today']['total_cost']:.4f}")
                    st.caption(f"Stage 1: ${cost_summary['today']['stage1_cost']:.4f}")
                    st.caption(f"Stage 2: ${cost_summary['today']['stage2_cost']:.4f}")

                with col2:
                    st.metric("Total Cost", f"${cost_summary['total']['total_cost']:.4f}")
                    st.caption(f"Professors: {cost_summary['total']['professors_processed']}")
                    st.caption(f"Emails: {cost_summary['total']['emails_generated']}")
            except Exception as e:
                st.error(f"Error loading cost data: {e}")

        st.markdown("---")

        # Database Management
        st.header("🗄️ Database Management")
        if st.session_state.orchestrator:
            if st.button("🧹 Clean Duplicates", help="Remove duplicate professor entries"):
                with st.spinner("Cleaning duplicate professors..."):
                    removed_count = st.session_state.orchestrator.db.clean_duplicate_professors()
                    if removed_count > 0:
                        st.success(f"✅ Removed {removed_count} duplicate professor(s)")
                    else:
                        st.info("ℹ️ No duplicates found")

        st.markdown("---")

        # CV Upload
        st.header("📄 CV Upload")
        cv_file = st.file_uploader("Upload your CV (PDF)", type=['pdf'])

        # Check if CV is already uploaded and analyzed
        cv_exists = os.path.exists("uploaded_cv.pdf")
        research_profile_file = "research_profile.txt"
        
        if cv_exists:
            st.success("✅ CV already uploaded")
            st.session_state.cv_uploaded = True
            
            # Also load the research profile if it exists
            if os.path.exists(research_profile_file) and not st.session_state.get('research_profile'):
                try:
                    with open(research_profile_file, 'r', encoding='utf-8') as f:
                        research_profile = f.read().strip()
                        if research_profile:
                            st.session_state.research_profile = research_profile
                            st.session_state.cv_analyzed = True
                            st.info("✅ Research profile loaded from previous analysis")
                except Exception as e:
                    st.warning(f"Could not load research profile: {e}")
            elif not os.path.exists(research_profile_file) and st.session_state.orchestrator:
                # CV exists but no research profile - offer to auto-analyze
                st.info("📝 CV found but research profile needs to be generated")
                if st.button("🔄 Generate Research Profile from Existing CV", type="primary"):
                    with st.spinner("Analyzing existing CV..."):
                        try:
                            # Read the existing CV file
                            with open("uploaded_cv.pdf", 'rb') as cv_file:
                                pdf_reader = PyPDF2.PdfReader(cv_file)
                                cv_text = ""
                                for page in pdf_reader.pages:
                                    cv_text += page.extract_text() + "\n"

                            # Generate research profile
                            research_profile_prompt = f"""
                            Based on the following CV content, create a comprehensive research profile:
                            
                            {cv_text[:3000]}
                            
                            Extract and summarize:
                            1. Research interests and expertise areas
                            2. Technical skills and programming languages
                            3. Academic background and achievements
                            4. Research experience and projects
                            5. Publications or notable work
                            6. Career goals and PhD interests
                            
                            Create a 200-word professional research profile that highlights alignment potential with professors.
                            """

                            response = st.session_state.orchestrator.api_manager.client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": research_profile_prompt}],
                                temperature=0.3,
                                max_tokens=400
                            )

                            research_profile = response.choices[0].message.content or ""
                            st.session_state.research_profile = research_profile
                            st.session_state.cv_analyzed = True
                            
                            # Save research profile to file for persistence
                            try:
                                with open("research_profile.txt", 'w', encoding='utf-8') as f:
                                    f.write(research_profile)
                                st.success("✅ Research profile generated and saved!")
                                st.rerun()
                            except Exception as e:
                                st.warning(f"Could not save research profile: {e}")
                                
                        except Exception as e:
                            st.error(f"❌ Error analyzing existing CV: {e}")
        
        if cv_file and st.session_state.orchestrator:
            # Only show analyze button if CV hasn't been analyzed or user wants to re-analyze
            if not st.session_state.get('cv_analyzed', False) or st.button("� Re-analyze CV"):
                if st.button("�📊 Analyze CV", key="analyze_cv_btn"):
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

                            # Generate research profile
                            research_profile_prompt = f"""
                            Analyze this CV and create a concise research profile for PhD applications:
                            
                            {cv_text[:3000]}
                            
                            Extract and summarize:
                            1. Research interests and expertise areas
                            2. Technical skills and programming languages
                            3. Academic background and achievements
                            4. Research experience and projects
                            5. Publications or notable work
                            6. Career goals and PhD interests
                            
                            Create a 200-word professional research profile that highlights alignment potential with professors.
                            """

                            response = st.session_state.orchestrator.api_manager.client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": research_profile_prompt}],
                                temperature=0.3,
                                max_tokens=400
                            )

                            st.session_state.research_profile = response.choices[0].message.content or ""
                            st.session_state.cv_uploaded = True
                            st.session_state.cv_analyzed = True
                            
                            # Save research profile to file for persistence
                            try:
                                with open("research_profile.txt", 'w', encoding='utf-8') as f:
                                    f.write(st.session_state.research_profile)
                            except Exception as e:
                                st.warning(f"Could not save research profile: {e}")
                            
                            st.success("✅ CV analyzed and research profile generated!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error analyzing CV: {e}")
            else:
                st.info("✅ CV has been analyzed. Research profile is ready!")
                
        # Show current research profile status
        if st.session_state.get('research_profile'):
            with st.expander("📝 Current Research Profile", expanded=False):
                st.text_area("Research Profile", value=st.session_state.research_profile, height=150, disabled=True)
                if st.button("🔄 Re-analyze CV", help="Regenerate research profile from CV"):
                    st.session_state.cv_analyzed = False
                    st.rerun()
        elif cv_file:
            st.warning("Please configure OpenAI API key first")

        # Gmail Status
        st.markdown("---")
        st.header("📧 Gmail Status")
        if st.session_state.orchestrator:
            gmail_status = "✅ Connected" if st.session_state.orchestrator.gmail_manager else "❌ Not Connected"
            if st.session_state.orchestrator.gmail_manager:
                st.markdown(f'<span class="gmail-status-connected">{gmail_status}</span>', unsafe_allow_html=True)
                st.caption(f"Email: {st.session_state.orchestrator.gmail_manager.user_email}")
            else:
                st.markdown(f'<span class="gmail-status-disconnected">{gmail_status}</span>', unsafe_allow_html=True)
                st.caption("Check credentials.json file")

        # User settings
        st.markdown("---")
        st.header("👤 User Settings")
        user_name = st.text_input("Your Name", value=os.getenv('USER_NAME', ''))
        user_email = st.text_input("Your Email", value=os.getenv('USER_EMAIL', ''))
        
        # Store in session state for validation
        if user_name:
            st.session_state.user_name = user_name
        if user_email:
            st.session_state.user_email = user_email

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
                            st.write(f"**Departments:** {row['Departments to Search']}")
                            st.write(f"**Notes:** {row['Notes']}")

                        with col_b:
                            include = st.checkbox("Include", key=f"include_{idx}", value=True)

                        with col_c:
                            if st.button("🗑️ Remove", key=f"remove_{idx}"):
                                df_targets = df_targets.drop(idx)
                                df_targets.to_csv("PhD_Targets.csv", index=False)
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
            st.warning("PhD_Targets.csv not found. Please add universities below.")

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
                            df_targets = pd.concat([df_targets, new_row], ignore_index=True)
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
            st.markdown('<div class="stage-indicator"><strong>Stage 1:</strong> Professor Discovery &amp; Matching (gpt-4o-mini - Cost Effective)</div>', unsafe_allow_html=True)
            st.markdown('<div class="stage-indicator"><strong>Stage 2:</strong> Personalized Email Generation (gpt-4 - High Quality)</div>', unsafe_allow_html=True)

            # Control buttons
            col_stage1, col_stage2, col_stop = st.columns(3)

            with col_stage1:
                if st.button("🔍 Run Stage 1", type="primary", help="Find and match professors (low cost)"):
                    if universities_to_research and st.session_state.get('research_profile'):
                        with st.spinner("Running Stage 1: Professor Discovery..."):
                            try:
                                research_profile = st.session_state.get('research_profile', '')
                                run_stage1_sync(universities_to_research, research_profile, st.session_state.orchestrator)
                                st.success("✅ Stage 1 completed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Stage 1 failed: {e}")
                                logger.error(f"Stage 1 error: {e}")
                    else:
                        st.warning("No universities selected or research profile missing!")

            with col_stage2:
                if st.button("📧 Run Stage 2", help="Generate quality emails (higher cost)"):
                    if st.session_state.get('user_name') and st.session_state.get('research_profile'):
                        with st.spinner("Running Stage 2: Email Generation..."):
                            try:
                                research_profile = st.session_state.get('research_profile', '')
                                user_name = st.session_state.get('user_name', '')
                                run_stage2_sync(research_profile, user_name, st.session_state.orchestrator)
                                st.success("✅ Stage 2 completed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Stage 2 failed: {e}")
                                logger.error(f"Stage 2 error: {e}")
                    else:
                        st.warning("Please enter your name and research profile in settings!")

            with col_stop:
                if st.button("⏹️ Stop", help="Stop current process"):
                    if st.session_state.orchestrator:
                        st.session_state.orchestrator.stop_research()
                        st.info("🛑 Process stopped")

            # Bulk Email Operations Section
            st.markdown("---")
            st.header("📧 Bulk Email Operations")
            
            # Email delay settings
            email_delay = st.slider("Email delay (seconds between emails)", 5, 60, 10, 5, help="Delay between emails to avoid rate limiting")
            
            # Bulk operation buttons
            bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
            
            with bulk_col1:
                if st.button("📝 Generate All Emails", type="secondary", help="Generate emails for all verified professors"):
                    if st.session_state.get('user_name') and st.session_state.get('research_profile'):
                        with st.spinner("Generating all emails..."):
                            try:
                                research_profile = st.session_state.get('research_profile', '')
                                user_name = st.session_state.get('user_name', '')
                                result = generate_bulk_emails_sync(research_profile, user_name, st.session_state.orchestrator)
                                
                                if result["success"]:
                                    st.success(f"✅ Generated {result['emails_generated']} emails (Cost: ${result.get('total_cost', 0):.4f})")
                                    if result.get('failed_generations'):
                                        st.warning(f"⚠️ {len(result['failed_generations'])} emails failed to generate")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Bulk generation failed: {result.get('error')}")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                                logger.error(f"Bulk generation error: {e}")
                    else:
                        st.warning("Please enter your name and research profile in settings!")

            with bulk_col2:
                # Handle Send All Drafted with proper state management
                if 'confirm_send_all' not in st.session_state:
                    st.session_state.confirm_send_all = False
                
                if not st.session_state.confirm_send_all:
                    if st.button("📤 Send All Drafted Emails", type="secondary", help="Send all emails that have been drafted"):
                        if st.session_state.get('user_name') and st.session_state.orchestrator and st.session_state.orchestrator.gmail_manager:
                            st.session_state.confirm_send_all = True
                            st.rerun()
                        elif not st.session_state.orchestrator or not st.session_state.orchestrator.gmail_manager:
                            st.warning("Gmail not configured!")
                        else:
                            st.warning("Please enter your name in settings!")
                else:
                    # Show confirmation dialog
                    st.warning("⚠️ This will send ALL drafted emails. Are you sure?")
                    col_confirm, col_cancel = st.columns(2)
                    
                    with col_confirm:
                        if st.button("✅ Confirm Send All", type="primary", key="confirm_send_yes"):
                            with st.spinner("Sending all drafted emails..."):
                                try:
                                    user_name = st.session_state.get('user_name', '')
                                    result = send_bulk_emails_sync(user_name, st.session_state.orchestrator, email_delay)
                                    
                                    if result["success"]:
                                        st.success(f"✅ Sent {result['emails_sent']}/{result['total_emails']} emails")
                                        if result.get('failed_sends'):
                                            st.warning(f"⚠️ {len(result['failed_sends'])} emails failed to send")
                                    else:
                                        st.error(f"❌ Bulk sending failed: {result.get('error')}")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                                    logger.error(f"Bulk sending error: {e}")
                                finally:
                                    st.session_state.confirm_send_all = False
                                    st.rerun()
                    
                    with col_cancel:
                        if st.button("❌ Cancel", key="send_cancel"):
                            st.session_state.confirm_send_all = False
                            st.rerun()

            with bulk_col3:
                # Handle Generate & Send All with proper state management
                if 'confirm_gen_send_all' not in st.session_state:
                    st.session_state.confirm_gen_send_all = False
                
                if not st.session_state.confirm_gen_send_all:
                    if st.button("🚀 Generate & Send All", type="primary", help="Generate emails for all verified professors and send immediately"):
                        if st.session_state.get('user_name') and st.session_state.get('research_profile') and st.session_state.orchestrator and st.session_state.orchestrator.gmail_manager:
                            st.session_state.confirm_gen_send_all = True
                            st.rerun()
                        elif not st.session_state.orchestrator or not st.session_state.orchestrator.gmail_manager:
                            st.warning("Gmail not configured!")
                        else:
                            st.warning("Please enter your name and research profile in settings!")
                else:
                    # Show confirmation dialog
                    st.warning("⚠️ This will generate AND send ALL emails automatically. Are you sure?")
                    col_confirm, col_cancel = st.columns(2)
                    
                    with col_confirm:
                        if st.button("✅ Confirm Generate & Send All", type="primary", key="confirm_yes"):
                            with st.spinner("Generating and sending all emails..."):
                                try:
                                    research_profile = st.session_state.get('research_profile', '')
                                    user_name = st.session_state.get('user_name', '')
                                    result = generate_and_send_all_sync(research_profile, user_name, st.session_state.orchestrator, email_delay)
                                    
                                    if result["success"]:
                                        st.success(f"✅ Generated {result['emails_generated']} and sent {result['emails_sent']} emails (Cost: ${result.get('total_cost', 0):.4f})")
                                        if result.get('failed_generations') or result.get('failed_sends'):
                                            failed_total = len(result.get('failed_generations', [])) + len(result.get('failed_sends', []))
                                            st.warning(f"⚠️ {failed_total} operations failed")
                                    else:
                                        st.error(f"❌ Generate & send failed: {result.get('error')}")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                                    logger.error(f"Generate & send error: {e}")
                                finally:
                                    st.session_state.confirm_gen_send_all = False
                                    st.rerun()
                    
                    with col_cancel:
                        if st.button("❌ Cancel", key="confirm_cancel"):
                            st.session_state.confirm_gen_send_all = False
                            st.rerun()

            # Email stats
            st.markdown("---")
            st.subheader("📋 Email Status Overview")
            
            # Get count of emails in different states
            try:
                conn = sqlite3.connect(st.session_state.orchestrator.db.db_path)
                
                email_stats = pd.read_sql_query('''
                    SELECT 
                        COUNT(CASE WHEN status = 'verified' AND (draft_email_subject IS NULL OR draft_email_subject = '') THEN 1 END) as need_emails,
                        COUNT(CASE WHEN status = 'email_drafted' THEN 1 END) as drafted,
                        COUNT(CASE WHEN status = 'email_sent' THEN 1 END) as sent
                    FROM professors 
                    WHERE alignment_score >= 6.0 AND email != ''
                ''', conn)
                
                conn.close()
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("📝 Need Emails", int(email_stats['need_emails'].iloc[0]) if not email_stats.empty else 0)
                with stats_col2:
                    st.metric("📧 Drafted", int(email_stats['drafted'].iloc[0]) if not email_stats.empty else 0)
                with stats_col3:
                    st.metric("✅ Sent", int(email_stats['sent'].iloc[0]) if not email_stats.empty else 0)
                
            except Exception as e:
                st.error(f"Error loading email stats: {e}")
                
        else:
            st.warning("⚠️ Please configure API key and upload CV first!")

    with col2:
        st.header("👨‍🏫 Research Results")

        if st.session_state.orchestrator:
            # Filter and display options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                status_filter = st.selectbox("Status Filter", ["All", "pending", "verified", "email_drafted", "email_sent"])
            with filter_col2:
                min_score = st.slider("Min Alignment Score", 0.0, 10.0, 6.0, 0.5)

            # Get professors from database
            try:
                conn = sqlite3.connect(st.session_state.orchestrator.db.db_path)
                
                if status_filter == "All":
                    query = "SELECT * FROM professors WHERE alignment_score >= ? ORDER BY alignment_score DESC"
                    professors_df = pd.read_sql_query(query, conn, params=[min_score])
                else:
                    query = "SELECT * FROM professors WHERE alignment_score >= ? AND status = ? ORDER BY alignment_score DESC"
                    professors_df = pd.read_sql_query(query, conn, params=[min_score, status_filter])
                
                conn.close()

                if not professors_df.empty:
                    st.write(f"**{len(professors_df)} professors found**")

                    # Display professors
                    for idx, prof in professors_df.iterrows():
                        alignment_color = "🟢" if prof['alignment_score'] >= 8 else "🟡" if prof['alignment_score'] >= 6 else "🔴"

                        # Status badge
                        status_class = f"status-{prof['status'].replace('_', '-')}"
                        status_display = prof['status'].replace('_', ' ').title()

                        with st.expander(f"{alignment_color} {prof['name']} - {prof['university']} (Score: {prof['alignment_score']:.1f})"):
                            col_info, col_actions = st.columns([2, 1])

                            with col_info:
                                st.markdown(f'<span class="professor-status {status_class}">{status_display}</span>', unsafe_allow_html=True)
                                st.write(f"**Department:** {prof['department']}")
                                st.write(f"**Email:** {prof['email']}")
                                if prof['research_interests']:
                                    research_text = prof['research_interests'][:200] + ("..." if len(prof['research_interests']) > 200 else "")
                                    st.write(f"**Research:** {research_text}")
                                if prof['collaboration_potential']:
                                    st.write(f"**Collaboration Potential:** {prof['collaboration_potential']}")
                                if prof['profile_url']:
                                    st.write(f"**Profile:** [{prof['profile_url']}]({prof['profile_url']})")
                                st.write(f"**Costs:** Stage 1: ${prof['stage1_cost']:.4f}, Stage 2: ${prof['stage2_cost']:.4f}")

                                # Show email draft if available
                                if prof['draft_email_subject']:
                                    st.markdown("**📧 Draft Email:**")
                                    st.text_input("Subject:", value=prof['draft_email_subject'], key=f"subject_{prof['id']}", disabled=True)
                                    
                                    # Format email body for display
                                    email_body = prof['draft_email_body']
                                    formatted_display = email_body.replace('\\n\\n', '\n\n').replace('\\n', '\n')
                                    display_body = formatted_display[:300] + ("..." if len(formatted_display) > 300 else "")
                                    
                                    st.text_area("Body:", value=display_body, key=f"body_{prof['id']}", height=120, disabled=True)

                            with col_actions:
                                # Action buttons based on status
                                if prof['status'] == 'verified':
                                    if st.button("📧 Generate Email", key=f"gen_email_{prof['id']}"):
                                        if st.session_state.get('user_name') and st.session_state.get('research_profile'):
                                            with st.spinner("Generating email..."):
                                                try:
                                                    research_profile = st.session_state.get('research_profile', '')
                                                    user_name = st.session_state.get('user_name', '')
                                                    success = generate_single_email_sync(prof['id'], user_name, st.session_state.orchestrator, research_profile)
                                                    if success:
                                                        st.success("✅ Email generated!")
                                                        st.rerun()
                                                    else:
                                                        st.error("❌ Failed to generate email")
                                                except Exception as e:
                                                    st.error(f"❌ Error: {e}")
                                        else:
                                            st.warning("Please enter your name and research profile in settings!")

                                elif prof['status'] == 'email_drafted':
                                    # Show email preview button
                                    if st.button("👀 Preview Email", key=f"preview_{prof['id']}"):
                                        show_email_preview(prof)

                                    # Send individual email
                                    if st.button("📤 Send Email", key=f"send_{prof['id']}", type="primary"):
                                        if st.session_state.orchestrator.gmail_manager and st.session_state.get('user_name'):
                                            with st.spinner("Sending email..."):
                                                try:
                                                    user_name = st.session_state.get('user_name', '')
                                                    success = send_single_email_sync(prof['id'], user_name, st.session_state.orchestrator)
                                                    if success:
                                                        st.success("✅ Email sent!")
                                                        st.rerun()
                                                    else:
                                                        st.error("❌ Failed to send email")
                                                except Exception as e:
                                                    st.error(f"❌ Error: {e}")
                                        elif not st.session_state.orchestrator.gmail_manager:
                                            st.warning("Gmail not configured!")
                                        else:
                                            st.warning("Please enter your name!")

                                    # Edit email button
                                    if st.button("✏️ Edit Email", key=f"edit_{prof['id']}"):
                                        show_email_preview(prof)

                                elif prof['status'] == 'email_sent':
                                    st.success("✅ Email Sent")
                                    if st.button("📬 View Details", key=f"details_{prof['id']}"):
                                        if prof.get('email_sent_at'):
                                            sent_time = prof['email_sent_at'][:19] if len(prof['email_sent_at']) > 19 else prof['email_sent_at']
                                            st.info(f"📬 Email sent on: {sent_time}")
                                        else:
                                            st.info("📬 Email sent (timestamp not available)")

                                # Delete button (available for all statuses)
                                if st.button("🗑️ Delete", key=f"delete_{prof['id']}"):
                                    try:
                                        conn = sqlite3.connect(st.session_state.orchestrator.db.db_path)
                                        cursor = conn.cursor()
                                        cursor.execute("DELETE FROM professors WHERE id = ?", (prof['id'],))
                                        conn.commit()
                                        conn.close()
                                        st.success("✅ Professor deleted!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting professor: {e}")
                else:
                    st.info("No professors found yet. Run Stage 1 to discover professors!")

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
                pending = status_counts[status_counts['status'] == 'pending']['count'].sum() if not status_counts.empty else 0
                st.metric("⏳ Pending", int(pending))

            with prog_col2:
                verified = status_counts[status_counts['status'] == 'verified']['count'].sum() if not status_counts.empty else 0
                st.metric("✅ Verified", int(verified))

            with prog_col3:
                drafted = status_counts[status_counts['status'] == 'email_drafted']['count'].sum() if not status_counts.empty else 0
                st.metric("📧 Emails Drafted", int(drafted))

            with prog_col4:
                sent = status_counts[status_counts['status'] == 'email_sent']['count'].sum() if not status_counts.empty else 0
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
        if hasattr(st.session_state.orchestrator, 'progress_messages'):
            st.subheader("🔄 Live Progress")

            # Show recent progress messages
            recent_messages = st.session_state.orchestrator.progress_messages[-10:]
            for message in recent_messages:
                st.markdown(f'<div class="progress-message">{message}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Additional custom styling
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
    
    .bulk-operation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .confirmation-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()        