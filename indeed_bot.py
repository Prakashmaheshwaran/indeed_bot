"""
Indeed Auto-Apply Bot
---------------------
Automates job applications on Indeed using Camoufox.

Usage:
  - Configure your search and Chrome settings in config.yaml
  - Run: python indeed_bot.py

Author: @meteor314 
License: MIT
"""
import yaml
import time
import json
import os
import random
import sys
import platform
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from camoufox.sync_api import Camoufox
import logging
from rapidfuzz import fuzz
from openai import OpenAI


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
camoufox_config = config.get("camoufox", {})
user_data_dir = camoufox_config.get("user_data_dir")
language = camoufox_config.get("language")

# Load timing configuration
timing_config = config.get("timing", {})
base_delays = {
    "page_load": timing_config.get("page_load", 3),
    "between_applications": timing_config.get("between_applications", 5),
    "element_wait": timing_config.get("element_wait", 2),
    "retry_delay": timing_config.get("retry_delay", 5),
    "human_like_variation": timing_config.get("human_like_variation", 0.3)
}

# Load question answering configuration
qa_config = config.get("question_answering", {})
qa_enabled = qa_config.get("enabled", False)
qa_mode = qa_config.get("mode", "hybrid")
qa_on_unknown = qa_config.get("on_unknown_question", "ai_then_skip")
qa_fuzzy_threshold = qa_config.get("fuzzy_match_threshold", 80)
openrouter_config = qa_config.get("openrouter", {})
resume_context = qa_config.get("resume_context", "")


def load_progress():
    """Load progress from progress.json file.
    This file tracks applied/failed jobs to prevent duplicates and stores statistics.
    It's essential for the bot to function properly but is git-ignored for privacy.
    """
    progress_file = "progress.json"
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


def save_progress(progress):
    """Save progress to progress.json file."""
    try:
        with open("progress.json", "w") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"Error saving progress: {e}")


def update_progress(progress, job_url, success):
    """Update progress tracking."""
    if success:
        if job_url not in progress["applied_jobs"]:
            progress["applied_jobs"].append(job_url)
            progress["stats"]["total_applied"] += 1
    else:
        if job_url not in progress["failed_jobs"]:
            progress["failed_jobs"].append(job_url)
            progress["stats"]["total_failed"] += 1

    progress["last_run"] = datetime.now().isoformat()


def print_progress_stats(progress, total_jobs):
    """Print current progress statistics."""
    applied = len(progress["applied_jobs"])
    failed = len(progress["failed_jobs"])
    remaining = total_jobs - applied - failed

    print("\n=== Progress Summary ===")
    print(f"Total jobs found: {total_jobs}")
    print(f"Successfully applied: {applied}")
    print(f"Failed applications: {failed}")
    print(f"Remaining to process: {remaining}")
    if total_jobs > 0:
        print(f"Success rate: {applied/total_jobs*100:.1f}%")
    else:
        print("Success rate: N/A")

    if progress["last_run"]:
        print(f"Last run: {progress['last_run']}")
    
    # Print question answering stats if available
    if "question_stats" in progress:
        q_stats = progress["question_stats"]
        print(f"\n--- Question Answering Stats ---")
        print(f"Total questions answered: {q_stats.get('total_answered', 0)}")
        print(f"From bank: {q_stats.get('from_bank', 0)}")
        print(f"From AI: {q_stats.get('from_ai', 0)}")
        print(f"Skipped: {q_stats.get('skipped', 0)}")
    
    print("=" * 25)


# ============================================================================
# Question Bank Management Functions
# ============================================================================

def load_question_bank():
    """Load question bank from questions_bank.json file."""
    bank_file = "questions_bank.json"
    if os.path.exists(bank_file):
        try:
            with open(bank_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading question bank: {e}")
    
    # Return default structure if file doesn't exist
    return {
        "questions": [],
        "stats": {
            "total_questions": 0,
            "total_answered_from_bank": 0,
            "total_answered_from_ai": 0,
            "total_skipped": 0
        }
    }


def save_question_bank(bank):
    """Save question bank to questions_bank.json file."""
    try:
        with open("questions_bank.json", "w", encoding="utf-8") as f:
            json.dump(bank, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving question bank: {e}")


def add_question_to_bank(bank, question, answer, source="ai"):
    """Add a new question-answer pair to the bank."""
    # Check if question already exists
    for q in bank["questions"]:
        if q["question"].lower().strip() == question.lower().strip():
            # Update existing question
            q["answer"] = answer
            q["last_used"] = datetime.now().isoformat()
            return
    
    # Extract keywords from question
    keywords = extract_keywords(question)
    
    # Add new question
    new_question = {
        "question": question,
        "answer": answer,
        "keywords": keywords,
        "match_count": 0,
        "last_used": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "source": source
    }
    
    bank["questions"].append(new_question)
    bank["stats"]["total_questions"] = len(bank["questions"])


def extract_keywords(text):
    """Extract important keywords from question text."""
    # Simple keyword extraction - remove common words
    common_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "should", "could", "may", "might", "can", "must", "you", "your"
    }

    words = text.lower().split()
    keywords = [w.strip("?,.:;!") for w in words if w.lower() not in common_words and len(w) > 2]
    return keywords[:10]  # Return top 10 keywords


def format_job_id(job_url):
    """Extract and format job ID for clean display (20-25 chars max)."""
    try:
        if 'jk=' in job_url:
            job_id = job_url.split('jk=')[-1].split('&')[0]
        else:
            job_id = job_url.split('/')[-1]

        # Truncate if too long
        if len(job_id) > 25:
            job_id = job_id[:22] + "..."
        return job_id
    except:
        return job_url[:25] if len(job_url) > 25 else job_url


def format_text(text, max_length=50):
    """Truncate text to max_length with ellipsis if needed."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


# ============================================================================
# Fuzzy Matching Functions
# ============================================================================

def find_matching_question(bank, question_text, threshold=80):
    """
    Find a matching question in the bank using fuzzy matching.
    Returns (matched_question, similarity_score) or (None, 0) if no match found.
    """
    best_match = None
    best_score = 0
    
    question_lower = question_text.lower().strip()
    
    for q in bank["questions"]:
        stored_question = q["question"].lower().strip()
        
        # Calculate fuzzy similarity score
        similarity = fuzz.ratio(question_lower, stored_question)
        
        # Also check keyword overlap for better matching
        question_keywords = set(extract_keywords(question_text))
        stored_keywords = set(q.get("keywords", []))
        
        if question_keywords and stored_keywords:
            keyword_overlap = len(question_keywords & stored_keywords) / len(question_keywords | stored_keywords)
            # Combine text similarity with keyword overlap
            combined_score = (similarity * 0.7) + (keyword_overlap * 100 * 0.3)
        else:
            combined_score = similarity
        
        if combined_score > best_score:
            best_score = combined_score
            best_match = q
    
    if best_score >= threshold:
        return best_match, best_score
    
    return None, 0


# ============================================================================
# AI Integration Functions (OpenRouter)
# ============================================================================

def ask_ai_for_answer(question_text, resume_ctx, logger):
    """
    Ask AI (via OpenRouter) for an answer to the question.
    Returns answer string or None if failed.
    """
    if not openrouter_config.get("enabled", False):
        logger.info("AI answering is disabled in config")
        return None
    
    api_key = openrouter_config.get("api_key", "")
    if not api_key or api_key == "your_api_key_here":
        logger.warning("OpenRouter API key not configured")
        return None
    
    model = openrouter_config.get("model", "anthropic/claude-3.5-sonnet")
    timeout = openrouter_config.get("timeout", 30)
    
    try:
        # Initialize OpenAI client with OpenRouter endpoint
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Create prompt for the AI
        prompt = f"""You are helping fill out a job application form. Based on the candidate's resume and the question asked, provide a concise and appropriate answer.

Resume/Profile:
{resume_ctx}

Question: {question_text}

Instructions:
- Provide a direct, concise answer suitable for a job application form
- Keep answers brief (1-3 sentences or a few words for simple questions)
- Be honest and align with the resume information
- For yes/no questions, answer with just "Yes" or "No" followed by brief context if needed
- For experience questions, provide specific numbers/years from the resume
- Return ONLY the answer text, no explanations or preamble

Answer:"""

        logger.info(f"Asking AI: {question_text}")
        
        # Call OpenRouter API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            timeout=timeout
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info(f"AI answered: {answer}")
        
        return answer
    
    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        return None


# ============================================================================
# Alert System Functions
# ============================================================================

def play_alert_sound():
    """Play a system beep sound (cross-platform)."""
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            os.system("afplay /System/Library/Sounds/Glass.aiff")
        elif system == "Windows":
            import winsound
            winsound.Beep(1000, 500)  # 1000 Hz for 500ms
        else:  # Linux
            os.system("beep -f 1000 -l 500 2>/dev/null || printf '\a'")
    except Exception as e:
        # Fallback to terminal bell
        print("\a", end="", flush=True)


def pause_for_manual_input(question_text):
    """
    Pause execution and beep to alert user for manual input.
    Returns user's answer or None if skipped.
    """
    print("\n" + "="*60)
    print("⚠️  MANUAL INPUT REQUIRED")
    print("="*60)
    print(f"Question: {question_text}")
    print("\nThe bot cannot answer this question automatically.")

    # Play alert sound 3 times
    for _ in range(3):
        play_alert_sound()
        time.sleep(0.5)

    print("\nOptions:")
    print("1. Enter your answer")
    print("2. Press ENTER to skip this application")

    user_input = input("\nYour answer: ").strip()

    if user_input:
        return user_input
    else:
        print("Skipping this application...")
        return None


def handle_captcha_detection(page, context="page"):
    """
    Detect and handle captcha/Cloudflare protection pages.
    Plays beep sound and waits for manual intervention.
    """
    try:
        # Check for common captcha indicators
        captcha_indicators = [
            'cloudflare', 'captcha', 'protection', 'challenge',
            'cf-browser-verification', 'cf-challenge',
            'recaptcha', 'hcaptcha', 'turnstile'
        ]

        page_content = page.content().lower()
        page_title = (page.title() or "").lower()
        current_url = page.url.lower()

        # Check if any captcha indicators are present
        captcha_detected = False
        for indicator in captcha_indicators:
            if indicator in page_content or indicator in page_title or indicator in current_url:
                captcha_detected = True
                break

        # Also check for specific captcha elements
        try:
            captcha_elements = page.query_selector_all('[class*="captcha"], [id*="captcha"], iframe[src*="recaptcha"], div[class*="challenge"]')
            if captcha_elements:
                captcha_detected = True
        except:
            pass

        if captcha_detected:
            print("\n" + "="*60)
            print("🚨 CAPTCHA/PROTECTION DETECTED")
            print("="*60)
            print(f"Context: {context}")
            print(f"URL: {current_url}")
            print(f"Title: {page.title()}")
            print("\n⚠️  Please solve the captcha/protection challenge manually.")
            print("The bot will wait until the page loads properly.")

            # Play alert sound multiple times
            for _ in range(5):
                play_alert_sound()
                time.sleep(1)

            print("\n⏳ Waiting for captcha to be solved...")
            print("Press Ctrl+C to stop waiting if needed.")

            # Wait until captcha is resolved (page content changes significantly)
            initial_content = page.content()
            start_time = time.time()

            while time.time() - start_time < 300:  # Wait up to 5 minutes
                time.sleep(2)
                current_content = page.content()

                # If content changed significantly, captcha might be solved
                if len(current_content) > len(initial_content) * 1.2:  # 20% more content
                    print("✅ Page content updated - captcha may be resolved")
                    break

            print("✅ Continuing with automation...")
            return True

    except Exception as e:
        print(f"Error during captcha detection: {e}")

    return False


# ============================================================================
# Question Detection and Answering Functions
# ============================================================================

def detect_question_fields(page):
    """
    Detect all question fields on the current page.
    Returns list of tuples: (element, question_text, field_type)
    """
    questions = []
    
    try:
        # Find all input fields, textareas, and select elements
        input_elements = page.query_selector_all('input[type="text"], input:not([type]), textarea')
        select_elements = page.query_selector_all('select')
        radio_groups = {}
        checkbox_elements = page.query_selector_all('input[type="checkbox"]')
        
        # Process text inputs and textareas
        for elem in input_elements:
            if not elem.is_visible():
                continue
            
            # Try to find associated label or question text
            question_text = None
            
            # Method 1: Check for associated label
            elem_id = elem.get_attribute("id")
            if elem_id:
                label = page.query_selector(f'label[for="{elem_id}"]')
                if label:
                    question_text = label.inner_text().strip()
            
            # Method 2: Check for nearby label
            if not question_text:
                try:
                    parent = elem.evaluate_handle("el => el.parentElement")
                    label = parent.query_selector("label")
                    if label:
                        question_text = label.inner_text().strip()
                except:
                    pass
            
            # Method 3: Check placeholder
            if not question_text:
                placeholder = elem.get_attribute("placeholder")
                if placeholder and len(placeholder) > 3:
                    question_text = placeholder
            
            # Method 4: Check aria-label
            if not question_text:
                aria_label = elem.get_attribute("aria-label")
                if aria_label:
                    question_text = aria_label
            
            if question_text:
                field_type = "textarea" if elem.evaluate("el => el.tagName").lower() == "textarea" else "text"
                questions.append((elem, question_text, field_type))
        
        # Process select dropdowns
        for elem in select_elements:
            if not elem.is_visible():
                continue
            
            question_text = None
            elem_id = elem.get_attribute("id")
            
            if elem_id:
                label = page.query_selector(f'label[for="{elem_id}"]')
                if label:
                    question_text = label.inner_text().strip()
            
            if not question_text:
                aria_label = elem.get_attribute("aria-label")
                if aria_label:
                    question_text = aria_label
            
            if question_text:
                questions.append((elem, question_text, "select"))
        
        # Process radio buttons (group by name)
        radio_elements = page.query_selector_all('input[type="radio"]')
        for elem in radio_elements:
            if not elem.is_visible():
                continue
            
            name = elem.get_attribute("name")
            if name and name not in radio_groups:
                # Find question text for this radio group
                question_text = None
                elem_id = elem.get_attribute("id")
                
                if elem_id:
                    label = page.query_selector(f'label[for="{elem_id}"]')
                    if label:
                        question_text = label.inner_text().strip()
                
                # Try to find fieldset legend
                if not question_text:
                    try:
                        fieldset = elem.evaluate_handle("el => el.closest('fieldset')")
                        legend = fieldset.query_selector("legend")
                        if legend:
                            question_text = legend.inner_text().strip()
                    except:
                        pass
                
                if question_text:
                    radio_groups[name] = (elem, question_text, "radio")
        
        questions.extend(radio_groups.values())
        
    except Exception as e:
        print(f"Error detecting question fields: {e}")
    
    return questions


def answer_question(page, element, question_text, field_type, bank, logger, progress):
    """
    Main function to answer a question using hybrid approach.
    Returns tuple: (success: bool, source: str) where source is "bank", "ai", or "manual"
    """
    logger.info(f"Processing question: {question_text}")
    
    answer = None
    answer_source = None
    
    # Step 1: Try to find answer in question bank (if mode allows)
    if qa_mode in ["stored_only", "hybrid"]:
        matched_q, score = find_matching_question(bank, question_text, qa_fuzzy_threshold)
        if matched_q:
            answer = matched_q["answer"]
            answer_source = "bank"
            matched_q["match_count"] = matched_q.get("match_count", 0) + 1
            matched_q["last_used"] = datetime.now().isoformat()
            logger.info(f"Found match in bank (score: {score:.1f}): {answer}")
            
            # Update stats
            bank["stats"]["total_answered_from_bank"] = bank["stats"].get("total_answered_from_bank", 0) + 1
            if "question_stats" not in progress:
                progress["question_stats"] = {"total_answered": 0, "from_bank": 0, "from_ai": 0, "skipped": 0}
            progress["question_stats"]["from_bank"] += 1
            progress["question_stats"]["total_answered"] += 1

            return True, "bank"
    
    # Step 2: If no match and mode allows AI, try AI
    if not answer and qa_mode in ["hybrid", "ai_only"]:
        logger.info("No match in bank, trying AI...")
        ai_answer = ask_ai_for_answer(question_text, resume_context, logger)
        
        if ai_answer:
            answer = ai_answer
            answer_source = "ai"
            
            # Save AI answer to bank for future use
            add_question_to_bank(bank, question_text, answer, source="ai")
            save_question_bank(bank)
            
            # Update stats
            bank["stats"]["total_answered_from_ai"] = bank["stats"].get("total_answered_from_ai", 0) + 1
            if "question_stats" not in progress:
                progress["question_stats"] = {"total_answered": 0, "from_bank": 0, "from_ai": 0, "skipped": 0}
            progress["question_stats"]["from_ai"] += 1
            progress["question_stats"]["total_answered"] += 1

            logger.info(f"AI provided answer: {answer}")
            return True, "ai"
    
    # Step 3: If still no answer, handle based on config
    if not answer:
        if qa_on_unknown == "pause_and_beep":
            logger.info("No answer found, pausing for manual input...")
            manual_answer = pause_for_manual_input(question_text)
            
            if manual_answer:
                answer = manual_answer
                answer_source = "manual"
                
                # Save manual answer to bank
                add_question_to_bank(bank, question_text, answer, source="manual")
                save_question_bank(bank)
                logger.info(f"Manual answer provided: {answer}")
                return True, "manual"
            else:
                logger.warning("User skipped this application")
                return False, "skipped"
        else:
            # Skip this question/application
            logger.warning(f"Unable to answer question: {format_text(question_text)}")

            # Update stats
            if "question_stats" not in progress:
                progress["question_stats"] = {"total_answered": 0, "from_bank": 0, "from_ai": 0, "skipped": 0}
            progress["question_stats"]["skipped"] += 1

            return False, "skipped"
    
    # Step 4: Fill in the answer based on field type
    try:
        if field_type == "text" or field_type == "textarea":
            element.fill(answer)
            logger.info(f"Filled {field_type} field with answer")
        
        elif field_type == "select":
            # Try to select option that best matches the answer
            options = element.query_selector_all("option")
            best_match = None
            best_score = 0
            
            for option in options:
                option_text = option.inner_text().strip()
                score = fuzz.ratio(answer.lower(), option_text.lower())
                if score > best_score:
                    best_score = score
                    best_match = option
            
            if best_match and best_score > 50:
                value = best_match.get_attribute("value")
                element.select_option(value=value)
                logger.info(f"Selected dropdown option: {best_match.inner_text()}")
            else:
                logger.warning(f"Could not find matching option for: {answer}")
                return False
        
        elif field_type == "radio":
            # Find all radio buttons with same name
            name = element.get_attribute("name")
            radios = page.query_selector_all(f'input[type="radio"][name="{name}"]')
            
            best_match = None
            best_score = 0
            
            for radio in radios:
                # Get label text for this radio
                radio_id = radio.get_attribute("id")
                label_text = ""
                
                if radio_id:
                    label = page.query_selector(f'label[for="{radio_id}"]')
                    if label:
                        label_text = label.inner_text().strip()
                
                if label_text:
                    score = fuzz.ratio(answer.lower(), label_text.lower())
                    if score > best_score:
                        best_score = score
                        best_match = radio
            
            if best_match and best_score > 50:
                best_match.check()
                logger.info(f"Selected radio button")
            else:
                logger.warning(f"Could not find matching radio option for: {answer}")
                return False
        
        time.sleep(0.5)  # Small delay after filling
        return True, answer_source

    except Exception as e:
        logger.error(f"Error filling answer: {e}")
        return False, "error"


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


def human_delay(base_delay, variation_percent=None):
    """Add human-like random variation to delays."""
    if variation_percent is None:
        variation_percent = base_delays["human_like_variation"]

    variation = base_delay * variation_percent
    delay = base_delay + random.uniform(-variation, variation)
    time.sleep(max(0.5, delay))  # Minimum 0.5s delay


def smart_wait_for_element(page, selector, timeout=None, retry_interval=None):
    """Wait for element with intelligent retry logic."""
    if timeout is None:
        timeout = base_delays["element_wait"] * 10
    if retry_interval is None:
        retry_interval = base_delays["element_wait"] * 0.5

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            element = page.query_selector(selector)
            if element and element.is_visible():
                return element
        except:
            pass
        time.sleep(retry_interval)
    return None


def click_and_wait(element, timeout=None):
    """Click element and wait with human-like delay."""
    if element:
        element.click()
        human_delay(timeout or base_delays["element_wait"])


def apply_to_job(browser, job_url, language, logger, progress, question_bank, max_retries=3):
    """Open a new tab, apply to the job, log the result, and close the tab."""
    # Extract job ID for cleaner logging
    job_id = job_url.split('jk=')[-1].split('&')[0] if 'jk=' in job_url else job_url.split('/')[-1]

    for attempt in range(max_retries):
        page = browser.new_page()
        try:
            short_job_id = format_job_id(job_url)
            print(f"📄 Loading job application page: {short_job_id}")
            page.goto(job_url)
            page.wait_for_load_state("domcontentloaded")
            human_delay(base_delays["page_load"])

            # Check for captcha/protection on job page
            handle_captcha_detection(page, f"job application: {short_job_id}")
            # Try to find the apply button using robust, language-agnostic selectors
            apply_btn = None
            apply_selectors = [
                # Indeed Apply specific selectors
                'button:has(span[class*="css-1ebo7dz"])',  # Indeed's specific class
                'button[class*="indeed-apply"]',
                '[data-testid*="apply"] button',
                # Language-specific text selectors
                'button:visible:has-text("Postuler")',
                'button:visible:has-text("Apply")',
                'button:visible:has-text("Jetzt bewerben")',  # German
                'button:visible:has-text("Postular")',       # Spanish
                'button:visible:has-text("Candidature")',   # French
                # Generic fallback selectors
                'a:visible:has-text("Apply")',
                'a:visible:has-text("Postuler")'
            ]

            for attempt_btn in range(20):
                for selector in apply_selectors:
                    try:
                        if ":has-text" in selector:
                            # Handle text-based selectors
                            apply_btn = page.query_selector(f'button:visible:has-text("{selector.split(":has-text(")[1].rstrip(")")})"')
                        else:
                            apply_btn = page.query_selector(selector)

                        if apply_btn and apply_btn.is_visible():
                            # Verify it's actually an apply button by checking text content
                            btn_text = (apply_btn.inner_text() or "").lower()
                            if any(keyword in btn_text for keyword in ["apply", "postuler", "bewerben", "postular", "candidature"]):
                                break
                    except Exception as e:
                        continue

                if apply_btn:
                    break
                time.sleep(0.5)

            # Final fallback: look for any prominent button
            if not apply_btn:
                try:
                    btns = page.query_selector_all('button:visible, a:visible')
                    for btn in btns:
                        try:
                            btn_text = (btn.inner_text() or "").lower()
                            aria_label = (btn.get_attribute("aria-label") or "").lower()

                            # Skip unwanted buttons
                            if any(skip in btn_text or skip in aria_label for skip in ["close", "cancel", "fermer", "annuler", "schließen"]):
                                continue

                            # Look for apply-related text
                            if any(keyword in btn_text or keyword in aria_label for keyword in ["apply", "postuler", "bewerben", "postular", "candidature", "jetzt"]):
                                apply_btn = btn
                                break
                        except:
                            continue
                except Exception as e:
                    print(f"Error in final fallback button search: {e}")
            if apply_btn:
                print(f"🎯 Found apply button, clicking...")
                click_and_wait(apply_btn, 5)
            else:
                short_job_id = format_job_id(job_url)
                print(f"❌ No apply button found for {short_job_id}")
                logger.warning(
                    f"No Indeed Apply button found for {short_job_id}")
                page.close()
                return False

            # add timeout for the wizard loop
            start_time = time.time()
            while True:
                if time.time() - start_time > 40:
                    short_job_id = format_job_id(job_url)
                    print(f"⏰ Application timed out for {short_job_id}")
                    logger.warning(
                        f"Timeout applying to {short_job_id}, closing tab and moving to next.")
                    break
                current_url = page.url
                # Resume step: select resume card if present
                resume_card = page.query_selector(
                    '[data-testid="FileResumeCardHeader-title"]')
                if resume_card:
                    print(f"📄 Found resume card, selecting...")
                    # Click the resume card (or its parent if needed)
                    try:
                        resume_card.click()
                    except Exception:
                        parent = resume_card.evaluate_handle(
                            'node => node.parentElement')
                        if parent:
                            parent.click()
                    time.sleep(1)
                    continuer_btn = None
                    btns = page.query_selector_all('button:visible')
                    for btn in btns:
                        text = (btn.inner_text() or "").lower()
                        if "continuer" in text or "continue" in text:
                            continuer_btn = btn
                            break
                    if continuer_btn:
                        print(f"🔄 Found continue button, clicking...")
                        click_and_wait(continuer_btn, 3)
                        print(f"⏭️  Continuing to next step...")
                        continue  # go to next step

                # QUESTION ANSWERING: Detect and answer any questions on this page
                if qa_enabled:
                    try:
                        print(f"🔍 Scanning for questions on application page...")
                        detected_questions = detect_question_fields(page)

                        if detected_questions:
                            print(f"📝 Found {len(detected_questions)} question(s) to answer")
                            logger.info(f"Detected {len(detected_questions)} question(s) on page")

                            all_answered = True
                            for i, (elem, q_text, f_type) in enumerate(detected_questions, 1):
                                short_q = format_text(q_text, 45)
                                print(f"   {i}. {short_q}")

                                # Skip if field already has a value
                                try:
                                    current_value = elem.input_value() if f_type in ["text", "textarea"] else None
                                    if current_value and len(current_value.strip()) > 0:
                                        print(f"   ✓ Skipping already filled field")
                                        logger.info(f"Skipping already filled field: {format_text(q_text)}")
                                        continue
                                except:
                                    pass

                                # Try to answer the question
                                print(f"   🔄 Answering question {i}...")
                                answered, source = answer_question(page, elem, q_text, f_type, question_bank, logger, progress)

                                if answered:
                                    if source == "ai":
                                        print(f"   ✅ Answered using AI")
                                    elif source == "bank":
                                        print(f"   ✅ Answered from bank")
                                    elif source == "manual":
                                        print(f"   ✅ Answered manually")
                                    else:
                                        print(f"   ✅ Answered successfully")
                                else:
                                    print(f"   ❌ Failed to answer")
                                    all_answered = False

                                    # If configured to skip on unknown questions
                                    if qa_on_unknown in ["skip_immediately", "ai_then_skip"]:
                                        print(f"   🚫 Skipping application due to unanswered question")
                                        logger.warning(f"Skipping application due to unanswered question: {format_text(q_text)}")
                                        page.close()
                                        return False

                            # Save question bank after answering questions
                            if detected_questions:
                                save_question_bank(question_bank)
                                save_progress(progress)
                                print(f"💾 Saved {len(detected_questions)} question(s) to bank")
                        else:
                            print(f"📋 No questions detected on this page")

                    except Exception as e:
                        print(f"❌ Error in question answering: {e}")
                        logger.error(f"Error in question answering: {e}")
                else:
                    print(f"📋 Question answering disabled, proceeding with application...")

                # Try to find a submit button with multiple fallback strategies
                print(f"🔍 Looking for submit/continue button...")
                submit_btn = None
                print(f"   Current URL: {page.url}")
                submit_selectors = [
                    # Language-specific submit texts
                    'button:visible:has-text("déposer ma candidature")',
                    'button:visible:has-text("soumettre")',
                    'button:visible:has-text("submit")',
                    'button:visible:has-text("apply")',
                    'button:visible:has-text("bewerben")',  # German
                    'button:visible:has-text("postular")',  # Spanish
                    'button:visible:has-text("invia")',     # Italian
                    'button:visible:has-text("solliciteren")', # Dutch
                    'button:visible:has-text("ansøg")',     # Danish
                    # Generic submit-related text
                    'button:visible:has-text("continue")',
                    'button:visible:has-text("continuer")',
                    'button:visible:has-text("weiter")',
                    'button:visible:has-text("siguiente")',
                    # Type-based selectors
                    'button[type="submit"]',
                    'input[type="submit"]'
                ]

                for selector in submit_selectors:
                    try:
                        if ":has-text" in selector:
                            # Extract text from selector for proper query
                            text = selector.split(":has-text(")[1].rstrip(")").strip('"')
                            submit_btn = page.query_selector(f'button:visible:has-text("{text}")')
                        else:
                            submit_btn = page.query_selector(selector)

                        if submit_btn and submit_btn.is_visible():
                            break
                    except Exception as e:
                        continue

                # Fallback: look through all visible buttons for submit-like text
                if not submit_btn:
                    try:
                        btns = page.query_selector_all('button:visible')
                        for btn in btns:
                            btn_text = (btn.inner_text() or "").lower()
                            if any(keyword in btn_text for keyword in [
                                "submit", "apply", "send", "finish", "complete",
                                "soumettre", "postuler", "terminer", "finaliser",
                                "bewerben", "senden", "absenden",
                                "postular", "enviar", "finalizar"
                            ]):
                                submit_btn = btn
                                break

                        # Final fallback: last visible button (often the primary action)
                        if not submit_btn and btns:
                            submit_btn = btns[-1]
                    except Exception as e:
                        print(f"Error finding submit button: {e}")
                if submit_btn:
                    print(f"✅ Found submit button, clicking...")
                    click_and_wait(submit_btn, 3)
                    print(f"🎉 Application submitted successfully!")
                    short_job_id = format_job_id(job_url)
                    logger.info(f"Applied successfully to {short_job_id}")
                    break

                # fallback: try to find a visible and enabled button to continue (other steps)
                print(f"🔄 Trying fallback button search...")
                btn = page.query_selector(
                    'button[type="button"]:not([aria-disabled="true"]), button[type="submit"]:not([aria-disabled="true"])')
                if btn:
                    print(f"✅ Found fallback button, clicking...")
                    click_and_wait(btn, 3)
                    if "confirmation" in page.url or "submitted" in page.url:
                        print(f"🎉 Application submitted successfully!")
                        short_job_id = format_job_id(job_url)
                        logger.info(f"Applied successfully to {short_job_id}")
                        break
                else:
                    short_job_id = format_job_id(job_url)
                    print(f"❌ No continue/submit button found for {short_job_id}")
                    logger.warning(
                        f"No continue/submit button found for {short_job_id}")
                    break
            page.close()
            return True
        except Exception as e:
            short_job_id = format_job_id(job_url)
            error_msg = str(e)[:80] + "..." if len(str(e)) > 80 else str(e)
            print(f"❌ Attempt {attempt + 1} failed: {error_msg}")
            logger.warning(f"Attempt {attempt + 1} failed for {short_job_id}: {e}")
            page.close()
            if attempt < max_retries - 1:
                wait_time = base_delays["retry_delay"] * (attempt + 1)  # Progressive delay
                print(f"🔄 Retrying in {wait_time} seconds... (attempt {attempt + 2}/{max_retries})")
                human_delay(wait_time)
                continue
            else:
                print(f"💥 Failed to apply to {short_job_id} after {max_retries} attempts")
                logger.error(f"Failed to apply to {short_job_id} after {max_retries} attempts: {e}")
                return False


def setup_logger():
    logger = logging.getLogger("indeed_apply")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("indeed_apply.log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


with Camoufox(user_data_dir=user_data_dir,
              persistent_context=True,
              viewport={'width': 1920, 'height': 1080}) as browser:
    logger = setup_logger()
    progress = load_progress()
    question_bank = load_question_bank()
    
    # Initialize question stats in progress if not present
    if "question_stats" not in progress:
        progress["question_stats"] = {
            "total_answered": 0,
            "from_bank": 0,
            "from_ai": 0,
            "skipped": 0
        }
    
    page = browser.new_page()
    page.goto("https://" + language + ".indeed.com")

    cookies = page.context.cookies()
    ppid_cookie = next(
        (cookie for cookie in cookies if cookie['name'] == 'PPID'), None)
    if not ppid_cookie:
        print("🔐 Token not found, please log in to Indeed first.")
        print("Redirecting to login page...")
        print("You need to restart the bot after logging in.")
        page.goto(
            "https://secure.indeed.com/auth?hl=" + language)
        print("⏳ Waiting for manual login...")

        # Check for captcha during login
        handle_captcha_detection(page, "login page")

        time.sleep(1000)  # wait for manual login
    else:
        print("Token found, proceeding with job search...")
        search_config = config.get("search", {})
        base_url = search_config.get("base_url", "")
        start = search_config.get("start", "")
        end = search_config.get("end", "")

        listURL = []
        i = start
        while i <= end:
            url = f"{base_url}&start={i}"
            listURL.append(url)
            i += 10

        all_job_links = []
        for url in listURL:
            print(f"🔍 Visiting search page: {url}")
            page.goto(url)
            page.wait_for_load_state("domcontentloaded")
            human_delay(base_delays["page_load"])

            # Check for captcha/protection
            handle_captcha_detection(page, "job search page")

            print("⏳ Waiting for page to load completely...")
            time.sleep(5)  # Reduced from 10 since we have captcha detection

            try:
                links = collect_indeed_apply_links(page, language)
                all_job_links.extend(links)
                print(f"Found {len(links)} Indeed Apply jobs on this page.")
            except Exception as e:
                print("Error extracting jobs:", e)
            time.sleep(5)

        # Filter out already processed jobs
        applied_jobs = progress["applied_jobs"]
        failed_jobs = progress["failed_jobs"]
        new_job_links = [job for job in all_job_links if job not in applied_jobs and job not in failed_jobs]

        print(f"Total Indeed Apply jobs found: {len(all_job_links)}")
        print(f"Already processed: {len(all_job_links) - len(new_job_links)}")
        print(f"New jobs to process: {len(new_job_links)}")

        for job_url in new_job_links:
            # Extract job ID from URL for cleaner display
            job_id = job_url.split('jk=')[-1].split('&')[0] if 'jk=' in job_url else job_url.split('/')[-1]
            print(f"🔄 Applying to job: {job_id}")
            success = apply_to_job(browser, job_url, language, logger, progress, question_bank)
            update_progress(progress, job_url, success)
            save_progress(progress)

            if success:
                print(f"✅ Successfully applied to {job_id}")
            else:
                print(f"❌ Failed to apply to {job_id}")
                logger.error(f"Failed to apply to {job_id}")

            human_delay(base_delays["between_applications"])

        # Save final question bank state
        save_question_bank(question_bank)
        
        print_progress_stats(progress, len(all_job_links))
