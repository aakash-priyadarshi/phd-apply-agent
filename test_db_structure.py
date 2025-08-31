#!/usr/bin/env python3
"""
Test script to verify database structure and cost tracking
"""
import sqlite3
import sys

def check_database_structure():
    """Check the database structure and see what columns exist."""
    try:
        # Connect to database
        conn = sqlite3.connect("phd_outreach.db")
        cursor = conn.cursor()
        
        print("=== DATABASE STRUCTURE CHECK ===")
        
        # Check professors table
        cursor.execute("PRAGMA table_info(professors)")
        professors_columns = cursor.fetchall()
        
        print("\nPROFESSORS TABLE COLUMNS:")
        for col in professors_columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Check cost_tracking table
        cursor.execute("PRAGMA table_info(cost_tracking)")
        cost_columns = cursor.fetchall()
        
        print("\nCOST_TRACKING TABLE COLUMNS:")
        for col in cost_columns:
            print(f"  {col[1]} ({col[2]})")
            
        # Check if there's any data
        cursor.execute("SELECT COUNT(*) FROM professors")
        prof_count = cursor.fetchone()[0]
        print(f"\nTotal professors in database: {prof_count}")
        
        cursor.execute("SELECT COUNT(*) FROM cost_tracking")
        cost_count = cursor.fetchone()[0]
        print(f"Total cost tracking records: {cost_count}")
        
        # Test the cost summary query
        print("\n=== TESTING COST SUMMARY QUERY ===")
        
        # Try the query from get_cost_summary
        try:
            cursor.execute('''
                SELECT SUM(stage1_cost), SUM(stage2_cost), SUM(total_cost),
                       SUM(professors_processed), SUM(emails_generated)
                FROM cost_tracking
            ''')
            result = cursor.fetchone()
            print(f"Cost tracking query result: {result}")
        except Exception as e:
            print(f"❌ Cost tracking query failed: {e}")
        
        # Try professor cost query  
        try:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_professors,
                    SUM(stage1_cost) as total_stage1_cost,
                    SUM(stage2_cost) as total_stage2_cost,
                    SUM(CASE WHEN status = 'Email Sent' THEN 1 ELSE 0 END) as emails_sent
                FROM professors
            ''')
            result = cursor.fetchone()
            print(f"Professor cost query result: {result}")
        except Exception as e:
            print(f"❌ Professor cost query failed: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if check_database_structure():
        print("\n✅ Database structure check completed successfully")
    else:
        print("\n❌ Database structure check failed")
        sys.exit(1)
