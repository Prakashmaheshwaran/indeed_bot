"""
Indeed Job Scraper
-----------------
Scrapes Indeed for jobs with "Indeed Apply" from multiple URLs and saves to temporary CSV.

Usage:
  - Configure URLs in config.yaml under search.urls
  - Run: python indeed_scrape.py

Author: @meteor314
License: MIT
"""
import yaml
import time
import json
import os
import csv
import random
from datetime import datetime
from typing import List, Dict, Any
from camoufox.sync_api import Camoufox
import logging


def load_config():
    """Load configuration from config.yaml file."""
    with open("data/config.yaml", "r") as f:
        return yaml.safe_load(f)


def setup_logger():
    """Set up logging for the scraper."""
    logger = logging.getLogger("indeed_scrape")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/indeed_bot.log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def collect_indeed_apply_links(page, language):
    """Collect all 'Indeed Apply' job links from the current search result page."""
    links = []

    # Try multiple selectors for job cards as Indeed may change their structure
    job_cards_selectors = [
        'div[data-testid="slider_item"]',
        'div[data-jk]',
        '.jobsearch-ResultsList div[data-testid]',
        '.job_seen_beacon',
        '[data-jk] .job_seen_beacon'
    ]

    for selector in job_cards_selectors:
        try:
            job_cards = page.query_selector_all(selector)
            if job_cards:
                print(f"Found {len(job_cards)} job cards using selector: {selector}")
                break
        except Exception as e:
            print(f"Selector {selector} failed: {e}")
            continue
    else:
        print("No job cards found with any selector")
        return links

    for card in job_cards:
        try:
            # Multiple ways to detect Indeed Apply buttons
            indeed_apply_selectors = [
                '[data-testid="indeedApply"]',
                '.indeed-apply-button',
                'button[class*="indeed-apply"]',
                'span[class*="indeed-apply"]'
            ]

            indeed_apply = None
            for apply_selector in indeed_apply_selectors:
                indeed_apply = card.query_selector(apply_selector)
                if indeed_apply:
                    break

            if indeed_apply:
                # Multiple ways to find job links
                link_selectors = [
                    'a.jcs-JobTitle',
                    'a[data-jk]',
                    'h2 a',
                    '.jobtitle a'
                ]

                link = None
                for link_selector in link_selectors:
                    link = card.query_selector(link_selector)
                    if link:
                        break

                if link:
                    job_url = link.get_attribute('href')
                    if job_url:
                        if job_url.startswith('/'):
                            job_url = f"https://{language}.indeed.com{job_url}"
                        elif not job_url.startswith('http'):
                            job_url = f"https://{language}.indeed.com{job_url}"
                        links.append(job_url)
        except Exception as e:
            print(f"Error processing job card: {e}")
            continue

    return links


def human_delay(base_delay, variation_percent=0.3):
    """Add human-like random variation to delays."""
    variation = base_delay * variation_percent
    delay = base_delay + random.uniform(-variation, variation)
    time.sleep(max(0.5, delay))


def extract_job_id(job_url):
    """Extract job ID from URL for cleaner display."""
    try:
        if 'jk=' in job_url:
            return job_url.split('jk=')[-1].split('&')[0]
        else:
            return job_url.split('/')[-1]
    except:
        return job_url


def save_jobs_to_csv(jobs, filename="csv/temp_jobs.csv"):
    """Save job list to temporary CSV file."""
    # Wipe existing file first
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['job_url', 'job_id', 'scraped_at']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for job in jobs:
            writer.writerow({
                'job_url': job,
                'job_id': extract_job_id(job),
                'scraped_at': datetime.now().isoformat()
            })

    print(f"💾 Saved {len(jobs)} jobs to {filename}")


def load_progress():
    """Load progress from progress.json file."""
    progress_file = "data/progress.json"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading progress: {e}")
    return {
        "applied_jobs": [],
        "failed_jobs": [],
        "last_run": None,
        "stats": {"total_applied": 0, "total_failed": 0}
    }


def main():
    """Main scraping function."""
    config = load_config()
    logger = setup_logger()
    progress = load_progress()

    # Get configuration
    search_config = config.get("search", {})
    urls = search_config.get("urls", [])
    start = search_config.get("start", 0)
    end = search_config.get("end", 10)

    camoufox_config = config.get("camoufox", {})
    user_data_dir = camoufox_config.get("user_data_dir")
    language = camoufox_config.get("language", "www")

    timing_config = config.get("timing", {})
    page_load_delay = timing_config.get("page_load", 3)

    if not urls:
        print("❌ No URLs configured in config.yaml under search.urls")
        return

    print(f"🚀 Starting Indeed job scraper...")
    print(f"📋 Will scrape from {len(urls)} URL(s)")

    # Set up browser
    with Camoufox(user_data_dir=user_data_dir, persistent_context=True) as browser:
        page = browser.new_page()
        page.goto("https://" + language + ".indeed.com")

        # Check for login requirement
        cookies = page.context.cookies()
        ppid_cookie = next(
            (cookie for cookie in cookies if cookie['name'] == 'PPID'), None)
        if not ppid_cookie:
            print("❌ Token not found, please log in to Indeed first.")
            print("🔐 Redirecting to login page...")
            print("🔄 You need to restart the scraper after logging in.")
            page.goto("https://secure.indeed.com/auth?hl=" + language)
            time.sleep(1000)  # wait for manual login
            return
        else:
            print("✅ Token found, proceeding with job scraping...")

        all_job_links = []
        total_pages_scraped = 0

        # Scrape from each URL
        for url_index, base_url in enumerate(urls, 1):
            print(f"\n🔍 Scraping URL {url_index}/{len(urls)}: {base_url[:80]}...")

            # Generate page URLs for this search
            list_urls = []
            i = start
            while i <= end:
                url = f"{base_url}&start={i}"
                list_urls.append(url)
                i += 10

            print(f"📄 Will scrape {len(list_urls)} pages for this search")

            # Scrape each page
            for page_url in list_urls:
                try:
                    print(f"🌐 Visiting: {page_url}")
                    page.goto(page_url)
                    page.wait_for_load_state("domcontentloaded")

                    print("⏳ Waiting for page to load...")
                    time.sleep(page_load_delay)

                    # Handle Cloudflare protection if present
                    print("🛡️ Waiting for Cloudflare protection (if any)...")
                    time.sleep(10)

                    # Collect Indeed Apply jobs
                    links = collect_indeed_apply_links(page, language)
                    all_job_links.extend(links)
                    print(f"✅ Found {len(links)} Indeed Apply jobs on this page")

                except Exception as e:
                    print(f"❌ Error scraping page {page_url}: {e}")
                    logger.error(f"Error scraping page {page_url}: {e}")

                time.sleep(5)  # Delay between pages

            total_pages_scraped += len(list_urls)

        # Remove duplicates
        unique_jobs = list(set(all_job_links))

        # Filter out already processed jobs
        applied_jobs = set(progress.get("applied_jobs", []))
        failed_jobs = set(progress.get("failed_jobs", []))
        new_jobs = [job for job in unique_jobs if job not in applied_jobs and job not in failed_jobs]

        print("\n📊 Scraping Summary:")
        print(f"   Total pages scraped: {total_pages_scraped}")
        print(f"   Total unique jobs found: {len(unique_jobs)}")
        print(f"   Already processed: {len(unique_jobs) - len(new_jobs)}")
        print(f"   New jobs to process: {len(new_jobs)}")

        # Save to temporary CSV (always wiped first)
        if new_jobs:
            save_jobs_to_csv(new_jobs)
            print(f"🎯 Ready to apply to {len(new_jobs)} new jobs!")
        else:
            print("📭 No new jobs found to process")

        # Update progress
        progress["last_run"] = datetime.now().isoformat()
        progress["stats"]["total_scraped"] = len(unique_jobs)

        try:
            with open("data/progress.json", "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"Error saving progress: {e}")


if __name__ == "__main__":
    main()
