#!/usr/bin/env python3
"""
Script đơn giản để seed data bằng cách call API.
Dùng khi các services đang chạy trong Docker.

Usage:
    python seed_data.py                    # Seed embeddings cho labels và intents cho feedbacks mới
    python seed_data.py --recompute        # Seed lại tất cả (bao gồm cả data cũ)
    python seed_data.py --labels-only      # Chỉ seed embeddings cho labels
    python seed_data.py --intents-only     # Chỉ seed intents cho feedbacks
"""
import argparse
import requests
import sys
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cấu hình - Đọc từ environment variables
LABEL_BACKEND_PORT = os.getenv('LABEL_BACKEND_PORT', '8001')
API_BASE_URL = f"http://localhost:{LABEL_BACKEND_PORT}/api/v1"


def print_banner():
    """In banner."""
    print("=" * 70)
    print("  🚀 SEED DATA SCRIPT - Intent Analysis System")
    print("=" * 70)
    print()


def print_section(title):
    """In tiêu đề section."""
    print()
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


def check_health():
    """Kiểm tra health của services."""
    print_section("Checking Services Health")
    
    try:
        # Check label-backend
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Label Backend Service: OK")
        else:
            print("❌ Label Backend Service: NOT OK")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to services: {e}")
        print(f"   Make sure Docker services are running: docker-compose ps")
        return False


def seed_label_embeddings():
    """Seed embeddings cho tất cả labels."""
    print_section("Seeding Label Embeddings")
    
    try:
        print("📡 Calling API: POST /admin/seed-label-embeddings")
        print("⏳ Processing... (This may take a few minutes)")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/admin/seed-label-embeddings",
            timeout=600  # 10 minutes timeout
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print()
            print("✅ SUCCESS!")
            print(f"   Total labels: {result.get('total', 0)}")
            print(f"   Processed: {result.get('processed', 0)}")
            print(f"   Failed: {result.get('failed', 0)}")
            print(f"   Time taken: {elapsed_time:.2f} seconds")
            return True
        else:
            print(f"❌ FAILED! Status code: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout! The operation took too long.")
        print("   This might happen if there are many labels to process.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def seed_feedback_intents(recompute=False):
    """Seed intents cho feedbacks."""
    print_section("Seeding Feedback Intents")
    
    try:
        mode = "all feedbacks (recompute)" if recompute else "new feedbacks only"
        print(f"📡 Calling API: POST /admin/seed-feedback-intents?recompute={recompute}")
        print(f"   Mode: {mode}")
        print("⏳ Processing... (This may take a few minutes)")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/admin/seed-feedback-intents",
            params={"recompute": recompute},
            timeout=600  # 10 minutes timeout
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print()
            print("✅ SUCCESS!")
            print(f"   Total feedbacks: {result.get('total', 0)}")
            print(f"   Processed: {result.get('processed', 0)}")
            print(f"   Failed: {result.get('failed', 0)}")
            print(f"   Time taken: {elapsed_time:.2f} seconds")
            return True
        else:
            print(f"❌ FAILED! Status code: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout! The operation took too long.")
        print("   This might happen if there are many feedbacks to process.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Seed data cho Intent Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python seed_data.py                    # Seed tất cả (labels + feedbacks mới)
  python seed_data.py --recompute        # Seed lại tất cả including data cũ
  python seed_data.py --labels-only      # Chỉ seed labels
  python seed_data.py --intents-only     # Chỉ seed intents cho feedbacks mới
  
Note: Services phải đang chạy trong Docker (docker-compose up -d)
        """
    )
    
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Recompute tất cả data (bao gồm cả data đã có cache)'
    )
    parser.add_argument(
        '--labels-only',
        action='store_true',
        help='Chỉ seed embeddings cho labels'
    )
    parser.add_argument(
        '--intents-only',
        action='store_true',
        help='Chỉ seed intents cho feedbacks'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host của label backend (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port của label backend (default: đọc từ LABEL_BACKEND_PORT trong .env)'
    )
    
    args = parser.parse_args()
    
    # Override API_BASE_URL if custom host/port provided
    global API_BASE_URL
    host = args.host
    port = args.port if args.port else LABEL_BACKEND_PORT
    API_BASE_URL = f"http://{host}:{port}/api/v1"
    
    # Print banner
    print_banner()
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API Base URL: {API_BASE_URL}")
    print()
    
    # Check health
    if not check_health():
        print()
        print("❌ Health check failed. Please start the services first:")
        print("   docker-compose up -d")
        sys.exit(1)
    
    # Determine what to seed
    seed_labels = not args.intents_only
    seed_intents = not args.labels_only
    
    success = True
    
    # Seed labels
    if seed_labels:
        if not seed_label_embeddings():
            success = False
            if not args.intents_only:
                print()
                print("⚠️  Warning: Label embedding failed. Intents seeding might not work properly.")
                print("   Continue anyway? (y/n): ", end="")
                answer = input().strip().lower()
                if answer != 'y':
                    sys.exit(1)
    
    # Seed intents
    if seed_intents:
        if not seed_feedback_intents(recompute=args.recompute):
            success = False
    
    # Summary
    print()
    print("=" * 70)
    if success:
        print("  ✅ ALL OPERATIONS COMPLETED SUCCESSFULLY!")
    else:
        print("  ⚠️  SOME OPERATIONS FAILED! Check the logs above.")
    print("=" * 70)
    print()
    print(f"🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print()
        print("⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

