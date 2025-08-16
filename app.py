#!/usr/bin/env python3
"""
Complete PhD Outreach Automation - Full 2-Stage System
Stage 1: Automatic Web Scraping + Professor Discovery (gpt-4o-mini)
Stage 2: Automatic Email Generation (gpt-4)
"""

import streamlit as st
import pandas as pd
import sqlite3
import asyncio
import json
import os
import time
from pathlib import Path
from datetime import datetime
import openai
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import threading
import queue
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

# Web scraping imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import aiohttp

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(
    page_title="PhD Outreach Automation - Complete System",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stage1-box {
        background: linear-gradient(90deg, #32CD32, #228B22);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #00FF00;
        font-weight: bold;
    }
    
    .stage2-box {
        background: linear-gradient(90deg, #4169E1, #0000CD);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #87CEEB;
        font-weight: bold;
    }
    
    .progress-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .professor-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class Professor:
    """Professor data structure."""
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
    """University target structure."""
    name: str
    country: str
    departments: str
    priority: str
    notes: str
    status: str = "pending"
    professors_found: int = 0
    last_scraped: str = ""


class DatabaseManager:
    """Enhanced database manager."""

    def __init__(self, db_path: str = "phd_outreach.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with complete schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Drop and recreate tables to ensure correct schema
        cursor.execute("DROP TABLE IF EXISTS professors")
        cursor.execute("DROP TABLE IF EXISTS cost_tracking")
        cursor.execute("DROP TABLE IF EXISTS progress_log")

        # Complete professors table
        cursor.execute('''
            CREATE TABLE professors (
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
            CREATE TABLE cost_tracking (
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

        # Progress logging table
        cursor.execute('''
            CREATE TABLE progress_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stage TEXT,
                university TEXT,
                message TEXT,
                status TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def add_professor(self, professor: Professor) -> int:
        """Add professor to database."""
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

    def get_professors(self, status: Optional[str] = None) -> List[Dict]:
        """Get professors from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status:
            cursor.execute(
                'SELECT * FROM professors WHERE status = ? ORDER BY alignment_score DESC', (status,))
        else:
            cursor.execute(
                'SELECT * FROM professors ORDER BY alignment_score DESC')

        columns = [description[0] for description in cursor.description]
        professors = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return professors

    def update_professor_status(self, professor_id: int, status: str, **kwargs):
        """Update professor status and other fields."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        update_fields = ['status = ?']
        values = [status]

        for key, value in kwargs.items():
            update_fields.append(f'{key} = ?')
            values.append(value)

        values.append(professor_id)

        cursor.execute(f'''
            UPDATE professors 
            SET {', '.join(update_fields)}
            WHERE id = ?
        ''', values)

        conn.commit()
        conn.close()

    def log_progress(self, stage: str, university: str, message: str, status: str):
        """Log progress to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO progress_log (timestamp, stage, university, message, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), stage, university, message, status))

        conn.commit()
        conn.close()


class APIManager:
    """Manages OpenAI API calls with cost tracking."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.total_cost = 0.0
        self.stage1_cost = 0.0
        self.stage2_cost = 0.0

        # Updated pricing (2024)
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

    def stage1_extract_and_match(self, page_content: str, user_research_profile: str) -> Dict[str, Any]:
        """Stage 1: Extract professor info and match (gpt-4o-mini)."""
        prompt = f"""
        Extract professor information from this faculty page content and match with user's research profile.
        
        Faculty Page Content:
        {page_content[:4000]}
        
        User's Research Profile:
        {user_research_profile}
        
        Extract up to 10 professors and for each provide:
        1. Name and basic info
        2. Email address if found
        3. Research areas/interests
        4. Alignment score (1-10) with user's profile
        5. Brief reasoning for collaboration potential
        
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
            return {"professors": [], "cost": 0, "success": False, "error": str(e)}

    def stage2_draft_email(self, professor: Professor, user_research_profile: str,
                           user_name: str) -> Dict[str, Any]:
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
            return {"success": False, "cost": 0, "error": str(e)}


class WebScraper:
    """Advanced web scraper for university faculty pages."""

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """Setup Chrome WebDriver."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(
                service=service, options=chrome_options)

        except Exception as e:
            st.error(f"WebDriver setup failed: {e}")
            self.driver = None

    def close_driver(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()

    def find_faculty_pages(self, university: str, departments: List[str]) -> List[str]:
        """Find faculty pages for a university."""
        faculty_urls = []

        if not self.driver:
            return faculty_urls

        try:
            for dept in departments:
                # Enhanced search terms
                search_terms = [
                    f"{university} {dept} faculty directory",
                    f"{university} {dept} professors",
                    f"{university} {dept} people",
                    f"site:{university.lower().replace(' ', '').replace('university', '').replace('institute', '').replace('technology', '')}.edu {dept} faculty"
                ]

                # Limit to avoid rate limiting
                for search_term in search_terms[:2]:
                    try:
                        search_url = f"https://www.google.com/search?q={search_term.replace(' ', '+')}"

                        self.driver.get(search_url)
                        time.sleep(2)

                        # Extract search result links
                        links = self.driver.find_elements(
                            By.CSS_SELECTOR, "a[href]")

                        for link in links[:5]:  # Top 5 results
                            try:
                                href = link.get_attribute("href")
                                if href and any(term in href.lower() for term in ["faculty", "people", "staff", "directory"]):
                                    if university.lower().replace(" ", "") in href.lower():
                                        faculty_urls.append(href)
                            except:
                                continue

                        if faculty_urls:
                            break  # Found pages, no need to continue

                    except Exception as e:
                        continue

                # Rate limiting
                time.sleep(3)

        except Exception as e:
            st.error(f"Error finding faculty pages: {e}")

        return list(set(faculty_urls))  # Remove duplicates

    def scrape_professors(self, faculty_url: str, university: str, department: str,
                          user_research_profile: str, db: DatabaseManager) -> List[Professor]:
        """Scrape professors from faculty page."""
        professors = []

        if not self.driver:
            return professors

        try:
            db.log_progress("Stage 1", university,
                            f"Scraping {faculty_url}", "in_progress")

            self.driver.get(faculty_url)
            time.sleep(5)  # Wait for page to load

            # Get page content
            page_content = self.driver.page_source
            soup = BeautifulSoup(page_content, 'html.parser')

            # Extract text content for AI analysis
            text_content = soup.get_text()

            # Use Stage 1 API to extract and match professors
            stage1_result = self.api_manager.stage1_extract_and_match(
                text_content, user_research_profile
            )

            if stage1_result["success"]:
                for prof_data in stage1_result["professors"]:
                    try:
                        professor = Professor(
                            name=prof_data.get("name", ""),
                            university=university,
                            department=department,
                            email=prof_data.get("email", ""),
                            research_interests=prof_data.get(
                                "research_interests", ""),
                            profile_url=prof_data.get("profile_url", ""),
                            alignment_score=prof_data.get(
                                "alignment_score", 0.0),
                            collaboration_potential=prof_data.get(
                                "collaboration_potential", ""),
                            status="verified" if prof_data.get(
                                "alignment_score", 0) >= 6.0 else "pending",
                            created_at=datetime.now().isoformat(),
                            last_verified=datetime.now().isoformat(),
                            stage1_cost=stage1_result["cost"] /
                            len(stage1_result["professors"])
                        )

                        if professor.name and professor.alignment_score >= 6.0:
                            professors.append(professor)
                    except Exception as e:
                        continue

                db.log_progress("Stage 1", university,
                                f"Found {len(professors)} professors", "success")
            else:
                db.log_progress(
                    "Stage 1", university, f"Failed: {stage1_result.get('error', 'Unknown error')}", "error")

            # Rate limiting
            time.sleep(3)

        except Exception as e:
            db.log_progress("Stage 1", university, f"Error: {str(e)}", "error")

        return professors


class ResearchOrchestrator:
    """Main orchestrator for automated research."""

    def __init__(self, openai_api_key: str):
        self.api_manager = APIManager(openai_api_key)
        self.web_scraper = WebScraper(self.api_manager)
        self.db = DatabaseManager()
        self.is_running = False
        self.progress_messages = []

    def add_progress_message(self, message: str):
        """Add progress message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.progress_messages.append(f"[{timestamp}] {message}")
        if len(self.progress_messages) > 20:
            self.progress_messages = self.progress_messages[-20:]

    def run_stage1_research(self, universities: List[University], user_research_profile: str):
        """Run Stage 1: Automated professor discovery."""
        self.is_running = True
        self.add_progress_message(
            "🚀 Starting Stage 1: Automated Professor Discovery")

        try:
            for i, university in enumerate(universities):
                if not self.is_running:
                    break

                self.add_progress_message(
                    f"🔍 Researching {university.name} ({i+1}/{len(universities)})")

                departments = [dept.strip()
                               for dept in university.departments.split(',')]

                # Find faculty pages
                faculty_pages = self.web_scraper.find_faculty_pages(
                    university.name, departments)
                self.add_progress_message(
                    f"📄 Found {len(faculty_pages)} faculty pages for {university.name}")

                university_professors = []

                # Scrape each faculty page
                for page_url in faculty_pages[:3]:  # Limit to top 3 pages
                    for dept in departments:
                        professors = self.web_scraper.scrape_professors(
                            page_url, university.name, dept, user_research_profile, self.db
                        )

                        # Save professors to database
                        for professor in professors:
                            try:
                                prof_id = self.db.add_professor(professor)
                                university_professors.append(professor)
                                self.add_progress_message(
                                    f"✅ Added: {professor.name} (Score: {professor.alignment_score:.1f})")
                            except Exception as e:
                                self.add_progress_message(
                                    f"❌ Error adding {professor.name}: {e}")

                # Update cost tracking
                total_stage1_cost = sum(
                    p.stage1_cost for p in university_professors)
                self.add_progress_message(
                    f"💰 Stage 1 cost for {university.name}: ${total_stage1_cost:.4f}")

                self.add_progress_message(
                    f"✅ Completed {university.name}: {len(university_professors)} professors")

                # Rate limiting between universities
                time.sleep(5)

        except Exception as e:
            self.add_progress_message(f"❌ Stage 1 error: {e}")
        finally:
            self.is_running = False
            self.add_progress_message(
                "✅ Stage 1 Complete: Professor discovery finished!")

    def run_stage2_email_generation(self, user_research_profile: str, user_name: str):
        """Run Stage 2: Automated email generation."""
        self.add_progress_message("📧 Starting Stage 2: Email Generation")

        try:
            # Get verified professors who need emails
            professors = self.db.get_professors("verified")
            pending_professors = [
                p for p in professors if not p.get('draft_email_subject')]

            if not pending_professors:
                self.add_progress_message(
                    "ℹ️ No professors need email generation")
                return

            self.add_progress_message(
                f"📧 Generating emails for {len(pending_professors)} professors")

            emails_generated = 0
            total_stage2_cost = 0.0

            for professor_data in pending_professors:
                if not self.is_running:
                    break

                # Convert to Professor object
                professor = Professor(
                    **{k: v for k, v in professor_data.items() if k != 'id'})

                self.add_progress_message(
                    f"✍️ Generating email for {professor.name}")

                # Stage 2: Generate personalized email
                email_result = self.api_manager.stage2_draft_email(
                    professor, user_research_profile, user_name
                )

                if email_result["success"]:
                    # Update professor with email draft
                    self.db.update_professor_status(
                        professor_data['id'], 'email_drafted',
                        draft_email_subject=email_result["subject"],
                        draft_email_body=email_result["body"],
                        stage2_cost=email_result["cost"]
                    )

                    emails_generated += 1
                    total_stage2_cost += email_result["cost"]

                    self.add_progress_message(
                        f"✅ Email generated for {professor.name}")
                else:
                    self.add_progress_message(
                        f"❌ Failed to generate email for {professor.name}")

                # Rate limiting
                time.sleep(1)

            self.add_progress_message(
                f"✅ Stage 2 Complete: {emails_generated} emails generated (${total_stage2_cost:.4f})")

        except Exception as e:
            self.add_progress_message(f"❌ Stage 2 error: {e}")

    def stop_research(self):
        """Stop the research pipeline."""
        self.is_running = False
        if self.web_scraper:
            self.web_scraper.close_driver()


# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'research_profile' not in st.session_state:
    st.session_state.research_profile = ""
if 'cv_uploaded' not in st.session_state:
    st.session_state.cv_uploaded = False
if 'progress_messages' not in st.session_state:
    st.session_state.progress_messages = []


def analyze_cv(cv_text, api_manager):
    """Analyze CV and generate research profile."""
    prompt = f"""
    Analyze this CV and create a concise research profile for PhD applications:
    
    {cv_text[:3000]}
    
    Extract:
    1. Primary research interests and expertise
    2. Technical skills and methodologies
    3. Key achievements and publications
    4. Research goals and focus areas
    
    Create a 150-word research profile that can be used for professor matching.
    """

    try:
        response = api_manager.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing CV: {e}")
        return ""


def main():
    """Main Streamlit application."""

    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0078d4, #005a9e); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
        <h1>🎓 PhD Outreach Automation - Complete System</h1>
        <p>Fully Automated Stage 1 (Discovery) + Stage 2 (Email Generation)</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # OpenAI Setup
        openai_key = st.text_input(
            "OpenAI API Key", type="password", value=os.getenv('OPENAI_API_KEY', ''))

        if openai_key:
            if st.session_state.orchestrator is None:
                st.session_state.orchestrator = ResearchOrchestrator(
                    openai_key)
                st.success("✅ System initialized!")

        st.markdown("---")

        # Cost tracking
        st.header("💰 Real-time Cost Tracking")
        if st.session_state.orchestrator:
            st.metric(
                "Stage 1 Cost", f"${st.session_state.orchestrator.api_manager.stage1_cost:.4f}")
            st.metric(
                "Stage 2 Cost", f"${st.session_state.orchestrator.api_manager.stage2_cost:.4f}")
            st.metric(
                "Total Cost", f"${st.session_state.orchestrator.api_manager.total_cost:.4f}")
        else:
            st.metric("Total Cost", "$0.0000")

        st.markdown("---")

        # CV Upload
        st.header("📄 CV Upload")
        cv_file = st.file_uploader("Upload your CV (PDF)", type=['pdf'])

        if cv_file and st.session_state.orchestrator:
            if st.button("📊 Analyze CV"):
                with st.spinner("Analyzing CV..."):
                    try:
                        pdf_reader = PyPDF2.PdfReader(BytesIO(cv_file.read()))
                        cv_text = ""
                        for page in pdf_reader.pages:
                            cv_text += page.extract_text() + "\n"

                        # Save CV
                        with open("uploaded_cv.pdf", "wb") as f:
                            cv_file.seek(0)
                            f.write(cv_file.read())

                        # Analyze CV
                        research_profile = analyze_cv(
                            cv_text, st.session_state.orchestrator.api_manager)
                        if research_profile:
                            st.session_state.research_profile = research_profile
                            st.session_state.cv_uploaded = True
                            st.success("✅ CV analyzed!")
                    except
