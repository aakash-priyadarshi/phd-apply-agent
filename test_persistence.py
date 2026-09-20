#!/usr/bin/env python3
"""
Test script to check CV and research profile persistence
"""
import os
from phd_agent.config import load_settings

def check_persistence_files():
    """Check what persistence files exist."""
    print("=== PERSISTENCE FILES CHECK ===")
    
    settings = load_settings()
    cv_file = settings.cv_path
    profile_file = settings.profile_path
    
    print(f"CV file exists: {cv_file.exists()}")
    if cv_file.exists():
        stat = os.stat(cv_file)
        print(f"  CV file size: {stat.st_size} bytes")
        print(f"  CV last modified: {stat.st_mtime}")
    
    print(f"Research profile file exists: {profile_file.exists()}")
    if profile_file.exists():
        with open(profile_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"  Profile length: {len(content)} characters")
    else:
        print("  No research profile file found - this means it needs to be created on first analysis")

if __name__ == "__main__":
    check_persistence_files()
