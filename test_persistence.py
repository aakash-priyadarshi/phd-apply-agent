#!/usr/bin/env python3
"""
Test script to check CV and research profile persistence
"""
import os

def check_persistence_files():
    """Check what persistence files exist."""
    print("=== PERSISTENCE FILES CHECK ===")
    
    cv_file = "uploaded_cv.pdf"
    profile_file = "research_profile.txt"
    
    print(f"CV file exists: {os.path.exists(cv_file)}")
    if os.path.exists(cv_file):
        stat = os.stat(cv_file)
        print(f"  CV file size: {stat.st_size} bytes")
        print(f"  CV last modified: {stat.st_mtime}")
    
    print(f"Research profile file exists: {os.path.exists(profile_file)}")
    if os.path.exists(profile_file):
        with open(profile_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"  Profile length: {len(content)} characters")
            print(f"  Profile preview: {content[:100]}...")
    else:
        print("  No research profile file found - this means it needs to be created on first analysis")

if __name__ == "__main__":
    check_persistence_files()
