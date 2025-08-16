#!/usr/bin/env python3
"""
Database Fix Script - PhD Outreach Automation
Fixes the missing database columns and creates proper schema.
"""

import sqlite3
import os
from datetime import datetime


def fix_database():
    """Fix database schema issues."""
    print("🔧 Fixing database schema...")

    # Connect to database
    conn = sqlite3.connect("phd_outreach.db")
    cursor = conn.cursor()

    # Get existing table info
    try:
        cursor.execute("PRAGMA table_info(professors)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"Current columns: {columns}")
    except:
        columns = []

    # Drop and recreate table with correct schema
    cursor.execute("DROP TABLE IF EXISTS professors")
    cursor.execute("DROP TABLE IF EXISTS cost_tracking")
    cursor.execute("DROP TABLE IF EXISTS sent_emails")

    # Create professors table with all required columns
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

    # Create cost tracking table
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

    # Create sent emails table
    cursor.execute('''
        CREATE TABLE sent_emails (
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

    # Add some sample data for testing
    cursor.execute('''
        INSERT INTO professors (
            name, university, department, email, research_interests, 
            alignment_score, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        "Dr. Sample Professor", "MIT", "Computer Science",
        "sample@mit.edu", "Machine Learning, AI, Neural Networks",
        8.5, "verified", datetime.now().isoformat()
    ))

    cursor.execute('''
        INSERT INTO professors (
            name, university, department, email, research_interests, 
            alignment_score, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        "Dr. Test Researcher", "Stanford", "AI Lab",
        "test@stanford.edu", "Natural Language Processing, Deep Learning",
        7.2, "pending", datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    print("✅ Database schema fixed!")
    print("✅ Sample professors added!")
    print("🎯 Database is now ready for Streamlit app")


if __name__ == "__main__":
    fix_database()
