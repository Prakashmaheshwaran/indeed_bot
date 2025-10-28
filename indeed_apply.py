"""
Indeed Job Applier
-----------------
Reads jobs from temp_jobs.csv and applies to them using the existing application logic.

Usage:
  - Run: python indeed_apply.py
  - Requires: temp_jobs.csv (created by indeed_scrape.py)

Author: @meteor314
License: MIT
"""
import yaml
import time
import json
import os
import csv
import random
import sys
import platform
import re
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from camoufox.sync_api import Camoufox
import logging
from rapidfuzz import fuzz
from openai import OpenAI


def load_config():
    """Load configuration from config.yaml file."""
    with open("data/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_jobs_from_csv(filename="csv/temp_jobs.csv"):
    """Load job URLs from CSV file."""
    jobs = []
    if not os.path.exists(filename):
        print(f"❌ CSV file {filename} not found. Run indeed_scrape.py first.")
        return jobs

    try:
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                jobs.append(row['job_url'])
        print(f"📋 Loaded {len(jobs)} jobs from {filename}")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")

    return jobs


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


def save_progress(progress):
    """Save progress to progress.json file."""
    try:
        with open("data/progress.json", "w") as f:
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
        print("\n--- Question Answering Stats ---")
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
    bank_file = "data/questions_bank.json"
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
        with open("data/questions_bank.json", "w", encoding="utf-8") as f:
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


# ============================================================================
# Fuzzy Matching Functions
# ============================================================================

def find_matching_question(bank, question_text, threshold=70):
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

        keyword_overlap = 0
        if question_keywords and stored_keywords:
            keyword_overlap = len(question_keywords & stored_keywords) / len(question_keywords | stored_keywords)
            # Combine text similarity with keyword overlap
            combined_score = (similarity * 0.6) + (keyword_overlap * 100 * 0.4)
        else:
            combined_score = similarity

        # Special handling for common question types
        if "linkedin" in question_lower and "linkedin" in stored_question:
            combined_score = max(combined_score, 85)  # Boost LinkedIn matches
        if "email" in question_lower and "email" in stored_question:
            combined_score = max(combined_score, 85)  # Boost email matches
        if "phone" in question_lower and "phone" in stored_question:
            combined_score = max(combined_score, 85)  # Boost phone matches

        if combined_score > best_score:
            best_score = combined_score
            best_match = q

    if best_score >= threshold:
        return best_match, best_score

    return None, 0


# ============================================================================
# AI Integration Functions (OpenRouter)
# ============================================================================

def ask_ai_for_answer(question_text, resume_ctx, logger, openrouter_config):
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
- For LinkedIn questions, provide the LinkedIn profile URL from the resume (look for linkedin.com URLs)
- For contact information, extract details from the resume context
- For phone number questions, look for phone numbers in the resume (format: +1234567890)
- For email questions, look for email addresses in the resume
- For location questions, use location from the resume
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


def ask_ai_for_answer_enhanced(question_text, resume_ctx, logger, openrouter_config):
    """
    Enhanced AI prompting with more specific instructions for contact information.
    Returns answer string or None if failed.
    """
    if not openrouter_config.get("enabled", False):
        return None

    api_key = openrouter_config.get("api_key", "")
    if not api_key or api_key == "your_api_key_here":
        return None

    model = openrouter_config.get("model", "anthropic/claude-3.5-sonnet")
    timeout = openrouter_config.get("timeout", 30)

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Enhanced prompt with more specific instructions
        prompt = f"""You are helping fill out a job application form. I need you to extract specific information from the resume below.

Resume/Profile:
{resume_ctx}

Question: {question_text}

IMPORTANT INSTRUCTIONS:
- Look carefully through the resume for the requested information
- For PHONE NUMBER questions, find any phone number in format +1234567890
- For EMAIL questions, find any email address like user@domain.com
- For LINKEDIN questions, find any linkedin.com URLs
- For LOCATION questions, find city/state/country information
- For NAME questions, find the person's full name
- If the information is not explicitly in the resume, you may infer it from context
- Return ONLY the specific information requested, nothing else
- If you can't find the information, return "Not available"

Answer:"""

        logger.info(f"Enhanced AI asking: {question_text}")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            timeout=timeout
        )

        answer = response.choices[0].message.content.strip()

        # Clean up the answer
        if answer.lower() in ["not available", "n/a", "none", ""]:
            return None

        logger.info(f"Enhanced AI answered: {answer}")
        return answer

    except Exception as e:
        logger.error(f"Error calling enhanced OpenRouter API: {e}")
        return None


def is_critical_question(question_text):
    """
    Determine if a question is critical (blocks application progress).
    Returns True if question is critical, False otherwise.
    """
    question_lower = question_text.lower()

    # Critical questions that typically block application submission
    critical_keywords = [
        "are you authorized to work",
        "do you require sponsorship",
        "do you have work authorization",
        "are you legally eligible to work",
        "visa status",
        "work permit",
        "green card",
        "citizenship",
        "immigration status",
        "do you need visa sponsorship",
        "current salary",
        "expected salary",
        "salary expectation",
        "minimum salary",
        "desired salary"
    ]

    # Check if any critical keywords are in the question
    for keyword in critical_keywords:
        if keyword in question_lower:
            return True

    # Questions with asterisks (*) are often required
    if "*" in question_text:
        return True

    # Questions with "required" in them
    if "required" in question_lower:
        return True

    return False


def verify_all_questions_answered(page, logger, openrouter_config, resume_context):
    """
    Use AI to verify if all questions on the current page are properly answered.
    Returns True if verification passes, False if potential issues found.
    """
    if not openrouter_config.get("enabled", False):
        logger.info("AI verification disabled - skipping")
        return True

    try:
        logger.info("Starting AI verification of page content...")

        # Get page content for AI analysis
        page_content = page.content()
        page_url = page.url

        # Create prompt for AI verification
        verification_prompt = f"""You are analyzing a job application page to verify if all required questions are properly answered.

Current URL: {page_url}

Page Content (HTML snippet):
{page_content[:4000]}{"..." if len(page_content) > 4000 else ""}

Instructions:
1. Analyze the page content for form fields, input boxes, or questions that appear to be unanswered
2. Look for:
   - Empty input fields
   - Unselected radio buttons or checkboxes
   - Dropdowns with "Please select" or similar default values
   - Required field indicators (*, required text, etc.)
   - Error messages about missing information
3. Check if there are any visible questions or form sections that appear incomplete

Return your analysis in this exact format:
VERIFICATION_RESULT: [TRUE/FALSE]
CONFIDENCE: [HIGH/MEDIUM/LOW]
ISSUES_FOUND: [Brief description of any issues found, or "None detected"]

Examples:
VERIFICATION_RESULT: FALSE
CONFIDENCE: HIGH
ISSUES_FOUND: Found empty salary expectation field marked as required

VERIFICATION_RESULT: TRUE
CONFIDENCE: MEDIUM
ISSUES_FOUND: None detected"""

        # Call AI for verification
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_config.get("api_key", ""),
        )

        response = client.chat.completions.create(
            model=openrouter_config.get("model", "anthropic/claude-3.5-sonnet"),
            messages=[{"role": "user", "content": verification_prompt}],
            timeout=openrouter_config.get("timeout", 30)
        )

        ai_response = response.choices[0].message.content.strip()
        logger.info(f"AI verification response: {ai_response}")

        # Parse AI response
        lines = ai_response.split('\n')
        result_line = next((line for line in lines if line.startswith('VERIFICATION_RESULT:')), None)
        confidence_line = next((line for line in lines if line.startswith('CONFIDENCE:')), None)
        issues_line = next((line for line in lines if line.startswith('ISSUES_FOUND:')), None)

        if result_line:
            result = result_line.split(':')[1].strip().upper() == 'TRUE'
            logger.info(f"AI verification result: {result}")

            if issues_line:
                issues = issues_line.split(':')[1].strip()
                logger.info(f"AI verification issues: {issues}")

            if confidence_line:
                confidence = confidence_line.split(':')[1].strip()
                logger.info(f"AI verification confidence: {confidence}")

            return result
        else:
            logger.warning("Could not parse AI verification response")
            return True  # Default to continuing if we can't parse

    except Exception as e:
        logger.error(f"Error during AI verification: {e}")
        return True  # Default to continuing if verification fails


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


# ============================================================================
# Question Detection and Answering Functions
# ============================================================================

def detect_question_fields(page):
    """
    Comprehensive question detection with extensive debugging and fallback methods.
    Works with Indeed's actual HTML structure, whatever it may be.
    Returns list of tuples: (element, question_text, field_type, confidence_score)
    """
    logger = logging.getLogger("indeed_apply")
    logger.info("=== STARTING COMPREHENSIVE QUESTION DETECTION ===")

    questions = []

    try:
        # STEP 1: DEBUG - Show us what's actually on the page
        logger.info("🔍 DEBUGGING: Analyzing page structure...")
        debug_page_structure(page, logger)

        # STEP 2: Try multiple detection strategies
        logger.info("🔍 Trying multiple detection strategies...")

        # Strategy 1: Accessibility Tree (NEW - MOST RELIABLE)
        questions = detect_questions_with_accessibility_tree(page, logger)

        # Strategy 2: Basic selectors (fallback if accessibility fails)
        if not questions:
            logger.info("📋 Accessibility tree found no questions, trying basic selectors...")
            questions = detect_with_basic_selectors(page, logger)

        # Strategy 3: If that fails, try broader selectors
        if not questions:
            logger.info("📋 Basic selectors failed, trying broader detection...")
            questions = detect_with_broader_selectors(page, logger)

        # Strategy 4: If that fails, try content-based detection
        if not questions:
            logger.info("📋 Broader selectors failed, trying content-based detection...")
            questions = detect_with_content_analysis(page, logger)

        # Strategy 5: If that fails, try iframe detection (Indeed might use iframes)
        if not questions:
            logger.info("📋 Content analysis failed, trying iframe detection...")
            questions = detect_in_iframes(page, logger)

        # Strategy 6: If all else fails, use AI to find elements
        if not questions:
            logger.info("📋 All detection methods failed, using AI assistance...")
            questions = detect_with_ai_assistance(page, logger)

        # LOGGING SUMMARY
        logger.info("=== QUESTION DETECTION SUMMARY ===")
        logger.info(f"Total questions identified: {len(questions)}")
        if questions:
            logger.info("✅ SUCCESS! Identified questions:")
            for i, (elem, q_text, f_type, conf, method) in enumerate(questions, 1):
                logger.info(f"  {i}. '{q_text}' (type: {f_type}, method: {method})")
        else:
            logger.warning("⚠️ NO QUESTIONS DETECTED - All detection methods failed")

        return questions

    except Exception as e:
        logger.error(f"❌ Error in comprehensive question detection: {e}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []


def detect_questions_with_accessibility_tree(page, logger):
    """
    Use Playwright's accessibility API to detect form fields.
    Most reliable method - sees page like assistive technologies do.
    """
    logger.info("🌲 Using Accessibility Tree detection...")
    questions = []
    
    try:
        # Get accessibility snapshot
        snapshot = page.accessibility.snapshot()
        
        if not snapshot:
            logger.warning("No accessibility snapshot available")
            return []
        
        # Recursively find form fields
        def find_form_fields(node, depth=0):
            fields = []
            if not node:
                return fields
            
            role = node.get('role', '')
            name = node.get('name', '').strip()
            
            # Identify form field roles
            form_roles = ['textbox', 'combobox', 'listbox', 'spinbutton', 
                          'searchbox', 'radio', 'checkbox']
            
            if role in form_roles and name:
                # Found a field with accessible name (question text)
                fields.append({
                    'role': role,
                    'name': name,
                    'value': node.get('value', ''),
                    'description': node.get('description', '')
                })
                logger.info(f"  Found {role}: '{name}'")
            
            # Recurse into children
            for child in node.get('children', []):
                fields.extend(find_form_fields(child, depth+1))
            
            return fields
        
        form_fields = find_form_fields(snapshot)
        logger.info(f"📋 Accessibility tree found {len(form_fields)} form fields")
        
        # Now match accessibility nodes to actual DOM elements
        for field_data in form_fields:
            question_text = field_data['name']
            role = field_data['role']
            
            # Map accessibility role to field type
            field_type_map = {
                'textbox': 'text',
                'searchbox': 'text',
                'combobox': 'select',
                'listbox': 'select',
                'spinbutton': 'number',
                'radio': 'radio',
                'checkbox': 'checkbox'
            }
            field_type = field_type_map.get(role, 'text')
            
            # Find the actual DOM element using the accessible name
            elem = find_element_by_accessible_name(page, question_text, role, logger)
            
            if elem:
                questions.append((elem, question_text, field_type, 0.95, "accessibility_tree"))
                logger.info(f"✅ Accessibility match: '{question_text}' (type: {field_type})")
            else:
                logger.warning(f"⚠️ Could not find DOM element for: '{question_text}'")
        
        return questions
        
    except Exception as e:
        logger.error(f"Error in accessibility tree detection: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return []


def find_element_by_accessible_name(page, accessible_name, role, logger):
    """
    Find the actual DOM element corresponding to an accessible name.
    """
    try:
        # Try multiple strategies to locate the element
        
        # Strategy 1: Find by aria-label (exact match)
        try:
            elem = page.query_selector(f'[aria-label="{accessible_name}"]')
            if elem and elem.is_visible():
                logger.debug(f"Found by aria-label: {accessible_name}")
                return elem
        except:
            pass
        
        # Strategy 2: Find by label text using locator
        try:
            # Use Playwright's get_by_label which is accessibility-aware
            elem = page.get_by_label(accessible_name, exact=True)
            if elem and elem.is_visible():
                logger.debug(f"Found by label text: {accessible_name}")
                return elem
        except:
            pass
        
        # Strategy 3: Find by label text (fuzzy)
        try:
            elem = page.get_by_label(accessible_name, exact=False)
            if elem and elem.is_visible():
                logger.debug(f"Found by fuzzy label: {accessible_name}")
                return elem
        except:
            pass
        
        # Strategy 4: Find by placeholder
        try:
            elem = page.query_selector(f'[placeholder="{accessible_name}"]')
            if elem and elem.is_visible():
                logger.debug(f"Found by placeholder: {accessible_name}")
                return elem
        except:
            pass
        
        # Strategy 5: Find input near text containing accessible name
        try:
            elems = page.query_selector_all('input, textarea, select')
            for elem in elems:
                if not elem.is_visible():
                    continue
                
                # Check nearby labels
                elem_id = elem.get_attribute('id')
                if elem_id:
                    label = page.query_selector(f'label[for="{elem_id}"]')
                    if label:
                        label_text = label.inner_text().strip()
                        if accessible_name in label_text or label_text in accessible_name:
                            logger.debug(f"Found by associated label: {accessible_name}")
                            return elem
        except:
            pass
        
        logger.debug(f"Could not find element for: {accessible_name}")
        return None
        
    except Exception as e:
        logger.debug(f"Error in find_element_by_accessible_name: {e}")
        return None


def looks_like_placeholder(text):
    """
    Determine if text looks like placeholder text vs. actual user input.
    Returns True if it looks like a placeholder (should not skip).
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Common placeholder patterns
    placeholder_patterns = [
        'enter your',
        'type here',
        'type your',
        'select',
        'choose',
        'click to',
        'example:',
        'e.g.',
        'placeholder',
        'yyyy-mm-dd',
        'mm/dd/yyyy',
        'dd/mm/yyyy',
        'ex:',
        'sample',
    ]
    
    for pattern in placeholder_patterns:
        if pattern in text_lower:
            return True
    
    # Very short text might be placeholder
    if len(text.strip()) < 2:
        return True
    
    return False


def debug_page_structure(page, logger):
    """Debug function to analyze what's actually on the page."""
    try:
        logger.info("🔍 DEBUG: Analyzing page structure...")

        # Get page URL for context
        page_url = page.url
        logger.info(f"📄 Page URL: {page_url}")

        # Try to find all forms first
        forms = page.query_selector_all('form')
        logger.info(f"📋 Found {len(forms)} form elements")

        # Look for common form-related elements
        elements_to_check = {
            'input': 'input',
            'textarea': 'textarea',
            'select': 'select',
            'button': 'button',
            'div[role="textbox"]': 'div[role="textbox"]',
            'div[contenteditable]': 'div[contenteditable="true"]',
            '*[class*="input"]': '*[class*="input"]',
            '*[class*="field"]': '*[class*="field"]',
            '*[id*="input"]': '*[id*="input"]',
            '*[id*="field"]': '*[id*="field"]'
        }

        for name, selector in elements_to_check.items():
            try:
                elements = page.query_selector_all(selector)
                if elements:
                    visible_count = sum(1 for elem in elements if elem.is_visible())
                    logger.info(f"📋 {name} ({selector}): {len(elements)} total, {visible_count} visible")
                    if visible_count > 0 and visible_count <= 5:  # Show details for first few
                        for i, elem in enumerate(elements[:3]):
                            if elem.is_visible():
                                attrs = get_element_attributes(elem)
                                logger.info(f"    Element {i+1}: {attrs}")
            except Exception as e:
                logger.debug(f"Error checking {name}: {e}")

        # Check for iframes (Indeed might use them)
        iframes = page.query_selector_all('iframe')
        logger.info(f"📋 Found {len(iframes)} iframe elements")
        for i, iframe in enumerate(iframes[:3]):  # Check first 3 iframes
            src = iframe.get_attribute('src') or 'no-src'
            logger.info(f"    Iframe {i+1}: src='{src}'")

        # Sample the page content to see what we're dealing with
        try:
            content = page.content()
            logger.info(f"📄 Page content length: {len(content)} characters")

            # Look for form-related keywords in content
            form_keywords = ['input', 'form', 'textarea', 'select', 'name=', 'id=', 'placeholder=']
            for keyword in form_keywords:
                count = content.lower().count(keyword)
                if count > 0:
                    logger.info(f"📋 Found '{keyword}': {count} occurrences")

        except Exception as e:
            logger.error(f"Error analyzing page content: {e}")

    except Exception as e:
        logger.error(f"Error in debug_page_structure: {e}")


def get_element_attributes(element):
    """Get key attributes from an element for debugging."""
    try:
        attrs = {}
        common_attrs = ['id', 'name', 'type', 'class', 'placeholder', 'value']

        for attr in common_attrs:
            value = element.get_attribute(attr)
            if value:
                attrs[attr] = value[:50] + '...' if len(str(value)) > 50 else str(value)

        return attrs
    except:
        return {'error': 'could not get attributes'}


def detect_with_basic_selectors(page, logger):
    """Strategy 1: Basic selectors (what we were doing before)."""
    logger.info("🔍 Strategy 1: Basic selectors...")
    questions = []

    try:
        # Find all visible input elements (text, email, tel, etc.)
        input_elements = []
        input_types = ['input[type="text"]', 'input[type="email"]', 'input[type="tel"]',
                      'input[type="number"]', 'input[type="password"]', 'input:not([type])']

        for input_type in input_types:
            try:
                elements = page.query_selector_all(input_type)
                for elem in elements:
                    if elem.is_visible():
                        input_elements.append(elem)
            except Exception as e:
                logger.debug(f"Error with selector {input_type}: {e}")

        logger.info(f"📋 Found {len(input_elements)} visible input elements")

        # Find all visible textareas
        textarea_elements = []
        try:
            textareas = page.query_selector_all('textarea')
            for elem in textareas:
                if elem.is_visible():
                    textarea_elements.append(elem)
        except Exception as e:
            logger.debug(f"Error finding textareas: {e}")

        logger.info(f"📋 Found {len(textarea_elements)} visible textarea elements")

        # Find all visible select dropdowns
        select_elements = []
        try:
            selects = page.query_selector_all('select')
            for elem in selects:
                if elem.is_visible():
                    select_elements.append(elem)
        except Exception as e:
            logger.debug(f"Error finding selects: {e}")

        logger.info(f"📋 Found {len(select_elements)} visible select elements")

        # Process inputs
        for i, elem in enumerate(input_elements, 1):
            question_text = generate_question_for_element(elem, page, logger, i)
            if question_text:
                field_type = "text"
                questions.append((elem, question_text, field_type, 0.9, "basic_input"))
                logger.info(f"✅ Added input question {i}: '{question_text}'")

        # Process textareas
        for i, elem in enumerate(textarea_elements, 1):
            question_text = generate_question_for_element(elem, page, logger, i)
            if question_text:
                questions.append((elem, question_text, "textarea", 0.9, "basic_textarea"))
                logger.info(f"✅ Added textarea question {i}: '{question_text}'")

        # Process selects
        for i, elem in enumerate(select_elements, 1):
            question_text = generate_question_for_element(elem, page, logger, i)
            if question_text:
                questions.append((elem, question_text, "select", 0.9, "basic_select"))
                logger.info(f"✅ Added select question {i}: '{question_text}'")

    except Exception as e:
        logger.error(f"Error in basic selectors: {e}")

    logger.info(f"📊 Strategy 1 found {len(questions)} questions")
    return questions


def detect_with_broader_selectors(page, logger):
    """Strategy 2: Try broader, more aggressive selectors."""
    logger.info("🔍 Strategy 2: Broader selectors...")
    questions = []

    try:
        # Try even more general selectors
        broader_selectors = [
            'input',  # All inputs
            '*[role="textbox"]',  # ARIA textboxes
            '*[contenteditable="true"]',  # Editable divs
            'div[class*="input"]',  # Divs that look like inputs
            'div[class*="field"]',  # Divs that look like fields
        ]

        for selector in broader_selectors:
            try:
                elements = page.query_selector_all(selector)
                logger.info(f"📋 Selector '{selector}': {len(elements)} total elements")

                for i, elem in enumerate(elements, 1):
                    if elem.is_visible():
                        # Check if this looks like a form field
                        if looks_like_form_field(elem):
                            question_text = generate_question_for_element(elem, page, logger, i, selector)
                            if question_text:
                                field_type = determine_field_type(elem)
                                questions.append((elem, question_text, field_type, 0.7, f"broad_{selector}"))
                                logger.info(f"✅ Added broad selector question: '{question_text}' (type: {field_type})")

            except Exception as e:
                logger.debug(f"Error with broad selector {selector}: {e}")

    except Exception as e:
        logger.error(f"Error in broader selectors: {e}")

    logger.info(f"📊 Strategy 2 found {len(questions)} questions")
    return questions


def detect_with_content_analysis(page, logger):
    """Strategy 3: Analyze page content for form-like structures."""
    logger.info("🔍 Strategy 3: Content analysis...")
    questions = []

    try:
        content = page.content()

        # Look for form-related patterns in the HTML
        form_patterns = [
            (r'<input[^>]*type=["\']([^"\']+)["\'][^>]*>', 'input'),
            (r'<textarea[^>]*>', 'textarea'),
            (r'<select[^>]*>', 'select'),
        ]

        for pattern, field_type in form_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            logger.info(f"📋 Found {len(matches)} {field_type} elements in content")

        # If we find form elements in content but not via selectors,
        # try to create synthetic questions
        if content.lower().count('<input') > 0 or content.lower().count('<textarea') > 0:
            logger.info("📋 Found form elements in HTML content, creating synthetic questions...")

            # Create some synthetic questions based on common Indeed fields
            common_indeed_fields = [
                "First Name", "Last Name", "Email", "Phone", "Address", "City", "State", "Zip Code",
                "Resume", "Cover Letter", "LinkedIn", "Portfolio", "GitHub", "Salary Expectation"
            ]

            for i, field_name in enumerate(common_indeed_fields[:5], 1):  # Limit to first 5
                # Try to find an element that might match this field
                found_element = find_element_for_field(page, field_name.lower(), logger)
                if found_element:
                    questions.append((found_element, field_name, "text", 0.6, "content_analysis"))
                    logger.info(f"✅ Added content analysis question: '{field_name}'")

    except Exception as e:
        logger.error(f"Error in content analysis: {e}")

    logger.info(f"📊 Strategy 3 found {len(questions)} questions")
    return questions


def detect_in_iframes(page, logger):
    """Strategy 4: Check iframes for form content."""
    logger.info("🔍 Strategy 4: Iframe detection...")
    questions = []

    try:
        iframes = page.query_selector_all('iframe')
        logger.info(f"📋 Found {len(iframes)} iframes")

        for i, iframe in enumerate(iframes, 1):
            try:
                # Skip recaptcha iframes
                src = iframe.get_attribute('src') or ''
                if 'recaptcha' in src:
                    logger.debug(f"📋 Skipping recaptcha iframe {i}")
                continue

                iframe_content = iframe.content_frame
                if iframe_content:
                    logger.info(f"📋 Checking iframe {i} content...")
                    # Recursively check iframe content
                    iframe_questions = detect_question_fields(iframe_content)
                    questions.extend(iframe_questions)
                    logger.info(f"📋 Iframe {i} had {len(iframe_questions)} questions")
                else:
                    logger.debug(f"📋 Iframe {i} has no content frame")

            except Exception as e:
                logger.debug(f"Error checking iframe {i}: {e}")

    except Exception as e:
        logger.error(f"Error in iframe detection: {e}")

    logger.info(f"📊 Strategy 4 found {len(questions)} questions")
    return questions


def detect_with_ai_assistance(page, logger):
    """Strategy 5: Use AI to find and describe form elements."""
    logger.info("🔍 Strategy 5: AI assistance...")
    questions = []

    try:
        # Get page content for AI analysis
        page_content = page.content()[:5000]  # Limit to first 5000 chars

        ai_prompt = f"""You are analyzing a job application webpage to find form fields.
        The page content (first 5000 characters) is:
        {page_content}

        Please identify ALL form fields, input boxes, text areas, dropdowns, etc.
        For each field you find, provide:
        - What type of field it is (input, textarea, select, etc.)
        - Any labels, placeholders, or nearby text that describes what should be entered
        - A confidence score (0.0-1.0) of how certain you are

        Format your response as:
        FIELD: [field description or label]
        TYPE: [input|textarea|select|radio|checkbox]
        CONFIDENCE: [0.0-1.0]

        Example:
        FIELD: Enter your full name
        TYPE: input
        CONFIDENCE: 0.9

        Return only the fields you find, one per block."""

        # Call AI if available
        try:
            from config import config
            openrouter_config = config.get("question_answering", {}).get("openrouter", {})
        except:
            openrouter_config = {"enabled": False}

        if openrouter_config.get("enabled", False):
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_config.get("api_key", ""),
            )

            response = client.chat.completions.create(
                model=openrouter_config.get("model", "anthropic/claude-3.5-sonnet"),
                messages=[{"role": "user", "content": ai_prompt}],
                timeout=15
            )

            ai_response = response.choices[0].message.content.strip()
            logger.info(f"📋 AI response: {ai_response}")

            # Parse AI response (this is simplified - in reality we'd need better parsing)
            # For now, just log what AI found
            if "FIELD:" in ai_response:
                logger.info("📋 AI found potential form fields in the content")
                # We could try to create synthetic questions here
                # But for now, we'll just note that AI found something

        else:
            logger.info("📋 AI assistance not available (API not configured)")

    except Exception as e:
        logger.error(f"Error in AI assistance: {e}")

    logger.info(f"📊 Strategy 5 found {len(questions)} questions")
    return questions


def looks_like_form_field(element):
    """Check if an element looks like a form field."""
    try:
        # Check for form-related attributes
        tag_name = element.evaluate("el => el.tagName").lower()
        if tag_name in ['input', 'textarea', 'select']:
            return True

        # Check for ARIA roles
        role = element.get_attribute('role')
        if role in ['textbox', 'combobox', 'listbox']:
            return True

        # Check for contenteditable
        contenteditable = element.get_attribute('contenteditable')
        if contenteditable == 'true':
            return True

        # Check for class names that suggest form fields
        class_name = element.get_attribute('class') or ''
        form_keywords = ['input', 'field', 'textbox', 'textarea', 'select']
        if any(keyword in class_name.lower() for keyword in form_keywords):
            return True

        return False

    except:
        return False


def generate_question_for_element(element, page, logger, index, method="unknown"):
    """Generate a question text for an element."""
    try:
        # Try multiple methods to find question text
        question_text = find_basic_question_text(element, page, logger)

        if question_text:
            return question_text

        # Try to get from attributes
        placeholder = element.get_attribute('placeholder')
        if placeholder and len(placeholder) > 2:
            return placeholder

        name = element.get_attribute('name')
        if name and name != 'no-name':
            return generate_question_from_name(name)

        # Fallback to generic question
        return f"Question {index}"

    except Exception as e:
        logger.debug(f"Error generating question for element {index}: {e}")
        return f"Question {index}"


def find_element_for_field(page, field_name, logger):
    """Try to find an element that might correspond to a specific field."""
    try:
        # Look for elements with matching placeholder, name, or nearby text
        elements = page.query_selector_all('input, textarea, select')

        for elem in elements:
            if elem.is_visible():
                # Check placeholder
                placeholder = elem.get_attribute('placeholder') or ''
                if field_name in placeholder.lower():
                    return elem

                # Check name attribute
                name = elem.get_attribute('name') or ''
                if field_name in name.lower():
                    return elem

                # Check nearby text
                try:
                    parent_text = elem.evaluate_handle("el => el.parentElement?.innerText || ''")
                    if parent_text and field_name in parent_text.lower():
                        return elem
                except:
                    pass

        return None

    except Exception as e:
        logger.debug(f"Error finding element for field {field_name}: {e}")
        return None


def determine_field_type(element):
    """Determine the field type of an element."""
    try:
        tag_name = element.evaluate("el => el.tagName").lower()

        if tag_name == 'textarea':
            return 'textarea'
        elif tag_name == 'select':
            return 'select'
        elif tag_name == 'input':
            input_type = element.get_attribute('type') or 'text'
            return input_type
        else:
            return 'text'  # Default fallback

    except:
        return 'text'


def fill_custom_form_element(element, answer, logger):
    """
    Fill a custom form element (like Indeed's div-based inputs).
    Returns True if successful, False otherwise.
    """
    try:
        logger.info("🔧 Attempting to fill custom form element...")

        # Method 1: Try to find an input element inside the div
        input_inside = element.query_selector('input, textarea')
        if input_inside:
            logger.info("📋 Found input element inside custom div, using standard fill")
            input_inside.fill(answer)
            return True

        # Method 2: Try to find contenteditable element
        contenteditable = element.query_selector('[contenteditable="true"]')
        if contenteditable:
            logger.info("📋 Found contenteditable element, using keyboard typing")
            contenteditable.click()
            time.sleep(0.2)
            try:
                page.keyboard.type(answer, delay=100)
                logger.info("✅ Successfully typed into contenteditable element")
                return True
            except Exception as kb_e:
                logger.debug(f"❌ Keyboard typing into contenteditable failed: {kb_e}")
                return False

        # Method 3: Try clicking the element and typing (like a user would)
        logger.info("📋 No standard input found, trying click-and-type method")
        element.click()

        # Wait a bit for any focus events
        time.sleep(0.2)

        # Try to type the answer using keyboard
        # This simulates user typing by pressing keys
        try:
            # Use page.keyboard to type the answer
            page.keyboard.type(answer, delay=100)  # 100ms delay between keystrokes
            logger.info("✅ Successfully typed answer using keyboard")
            return True
        except Exception as kb_e:
            logger.debug(f"❌ Keyboard typing failed: {kb_e}")

            # Fallback: try using evaluate to set value via JavaScript
            try:
                element.evaluate(f"arguments[0].textContent = '{answer}';")
                logger.info("✅ Successfully set answer using JavaScript")
                return True
            except Exception as js_e:
                logger.debug(f"❌ JavaScript setting failed: {js_e}")

        return False

    except Exception as e:
        logger.error(f"❌ Error filling custom form element: {e}")
        return False


def fill_custom_select_element(element, answer, logger):
    """
    Fill a custom select element (like Indeed's div-based dropdowns).
    Returns True if successful, False otherwise.
    """
    try:
        logger.info("🔧 Attempting to fill custom select element...")

        # Method 1: Try to find a select element inside the div
        select_inside = element.query_selector('select')
        if select_inside:
            logger.info("📋 Found select element inside custom div, using standard select")
            return fill_standard_select(select_inside, answer, logger)

        # Method 2: Look for dropdown-like elements (buttons, divs with options)
        # This is more complex and may need to be customized based on Indeed's structure
        logger.info("📋 Looking for dropdown options in custom select...")

        # Try to find clickable elements that might be options
        clickable_options = element.query_selector_all('button, [role="option"], div[class*="option"]')
        if clickable_options:
            logger.info(f"📋 Found {len(clickable_options)} potential dropdown options")

            # Try to match the answer to option text
            best_match = None
            best_score = 0

            for option in clickable_options:
                try:
                    option_text = option.inner_text().strip()
                    score = fuzz.ratio(answer.lower(), option_text.lower())
                    logger.info(f"Option: '{option_text}' - Score: {score}")

                    if score > best_score and score > 50:  # Require higher confidence for custom selects
                        best_score = score
                        best_match = option
                except:
                    continue

            if best_match:
                logger.info(f"✅ Clicking dropdown option: '{best_match.inner_text().strip()}' (score: {best_score})")
                best_match.click()
                return True

        # Method 3: If all else fails, try typing the answer (some "selects" might be text inputs)
        logger.info("📋 No clickable options found, trying to type answer")
        element.click()
        time.sleep(0.2)
        try:
            page.keyboard.type(answer, delay=100)
            logger.info("✅ Successfully typed answer into custom select")
            return True
        except Exception as kb_e:
            logger.debug(f"❌ Keyboard typing into custom select failed: {kb_e}")
            return False

    except Exception as e:
        logger.error(f"❌ Error filling custom select element: {e}")
        return False


def fill_standard_select(element, answer, logger):
    """Fill a standard select element."""
    try:
        options = element.query_selector_all("option")
        logger.info(f"📋 Found {len(options)} dropdown options")
        best_match = None
        best_score = 0

        for option in options:
            option_text = option.inner_text().strip()
            score = fuzz.ratio(answer.lower(), option_text.lower())
            logger.info(f"Option: '{option_text}' - Score: {score}")
            if score > best_score:
                best_score = score
                best_match = option

        if best_match and best_score > 50:
            value = best_match.get_attribute("value")
            option_text = best_match.inner_text().strip()
            element.select_option(value=value)
            logger.info(f"✅ Selected dropdown option: '{option_text}' (score: {best_score})")
            return True
        else:
            logger.warning(f"❌ Could not find matching option for: '{answer}' (best score: {best_score})")
            return False

    except Exception as e:
        logger.error(f"❌ Error filling standard select: {e}")
        return False


def find_basic_question_text(element, page, logger):
    """Find question text using very basic methods for simple HTML structures."""
    try:
        # Method 1: Direct label association (if it exists)
        elem_id = element.get_attribute("id")
        if elem_id:
            label = page.query_selector(f'label[for="{elem_id}"]')
            if label and label.is_visible():
                return label.inner_text().strip()

        # Method 2: Parent element text (look for nearby text)
        try:
            parent = element.evaluate_handle("el => el.parentElement")
            if parent:
                parent_text = parent.inner_text().strip()
                # Look for question-like text in parent (shorter than 200 chars, contains ? or keywords)
                if (len(parent_text) > 5 and len(parent_text) < 200 and
                    ('?' in parent_text or
                     any(word in parent_text.lower() for word in ['name', 'email', 'phone', 'address', 'city', 'state', 'zip']))):
                    return parent_text
        except:
            pass

        # Method 3: Previous sibling text
        try:
            prev_sibling = element.evaluate_handle("el => el.previousElementSibling")
            if prev_sibling:
                prev_text = prev_sibling.inner_text().strip()
                if (len(prev_text) > 3 and len(prev_text) < 100 and
                    ('?' in prev_text or ':' in prev_text or
                     any(word in prev_text.lower() for word in ['name', 'email', 'phone', 'address']))):
                    return prev_text
        except:
            pass

        # Method 4: Placeholder text (already handled in main function)
        placeholder = element.get_attribute("placeholder")
        if placeholder and len(placeholder) > 3:
            return placeholder

    except Exception as e:
        logger.debug(f"Error finding question text: {e}")

    return None


def generate_question_from_name(field_name):
    """Generate a readable question from a field name."""
    # Convert camelCase or snake_case to readable text
    words = []
    current_word = ""

    for char in field_name:
        if char.isupper() and current_word:
            words.append(current_word.lower())
            current_word = char.lower()
        elif char == '_' or char == '-':
            if current_word:
                words.append(current_word.lower())
                current_word = ""
        else:
            current_word += char

    if current_word:
        words.append(current_word.lower())

    # Create a question from the words
    if words:
        # Capitalize first word and add question mark
        question = ' '.join(words).capitalize()
        if not question.endswith('?'):
            question += "?"
        return question

    return field_name  # Fallback to original name


def find_question_text_for_element(element, page, logger, is_radio_group=False):
    """
    Enhanced question text detection for a single element.
    Returns (question_text, confidence_score, detection_method)
    """
    methods_tried = []
    question_text = None
    confidence = 0.0

    # Method 1: Direct label association (highest confidence)
    elem_id = element.get_attribute("id")
    if elem_id:
        methods_tried.append("label_for_id")
        label = page.query_selector(f'label[for="{elem_id}"]')
        if label and label.is_visible():
            question_text = label.inner_text().strip()
            confidence = 0.9
            logger.info(f"Method 1 (label_for_id) found: '{question_text}' (confidence: {confidence})")
            return question_text, confidence, "label_for_id"

    # Method 2: Parent element labels
    try:
        methods_tried.append("parent_label")
        parent = element.evaluate_handle("el => el.parentElement")
        if parent:
            label = parent.query_selector("label")
            if label and label.is_visible():
                question_text = label.inner_text().strip()
                confidence = 0.8
                logger.info(f"Method 2 (parent_label) found: '{question_text}' (confidence: {confidence})")
                return question_text, confidence, "parent_label"
    except Exception as e:
        logger.debug(f"Method 2 failed: {e}")

    # Method 3: Preceding sibling text (question-like)
    try:
        methods_tried.append("preceding_text")
        prev_sibling = element.evaluate_handle("el => el.previousElementSibling")
        if prev_sibling:
            prev_text = prev_sibling.inner_text().strip()
            if (prev_text and len(prev_text) > 5 and
                ('?' in prev_text or
                 any(word in prev_text.lower() for word in ['do you', 'are you', 'have you', 'what', 'how', 'when', 'where', 'why', 'please', 'select']))):
                question_text = prev_text
                confidence = 0.7
                logger.info(f"Method 3 (preceding_text) found: '{question_text}' (confidence: {confidence})")
                return question_text, confidence, "preceding_text"
    except Exception as e:
        logger.debug(f"Method 3 failed: {e}")

    # Method 4: Aria labels and placeholders
    methods_tried.append("aria_placeholder")
    placeholder = element.get_attribute("placeholder")
    aria_label = element.get_attribute("aria-label")
    aria_describedby = element.get_attribute("aria-describedby")

    # Try aria-describedby first (points to description element)
    if aria_describedby:
        try:
            desc_element = page.query_selector(f'#{aria_describedby}')
            if desc_element:
                question_text = desc_element.inner_text().strip()
                confidence = 0.75
                logger.info(f"Method 4 (aria-describedby) found: '{question_text}' (confidence: {confidence})")
                return question_text, confidence, "aria-describedby"
        except:
            pass

    # Try aria-label
    if aria_label and len(aria_label) > 3:
        question_text = aria_label
        confidence = 0.6
        logger.info(f"Method 4 (aria-label) found: '{question_text}' (confidence: {confidence})")
        return question_text, confidence, "aria-label"

    # Try placeholder
    if placeholder and len(placeholder) > 3:
        question_text = placeholder
        confidence = 0.5
        logger.info(f"Method 4 (placeholder) found: '{question_text}' (confidence: {confidence})")
        return question_text, confidence, "placeholder"

    # Method 5: For radio groups, check fieldset legend
    if is_radio_group:
        try:
            methods_tried.append("fieldset_legend")
            fieldset = element.evaluate_handle("el => el.closest('fieldset')")
            if fieldset:
                legend = fieldset.query_selector("legend")
                if legend and legend.is_visible():
                    question_text = legend.inner_text().strip()
                    confidence = 0.8
                    logger.info(f"Method 5 (fieldset_legend) found: '{question_text}' (confidence: {confidence})")
                    return question_text, confidence, "fieldset_legend"
        except Exception as e:
            logger.debug(f"Method 5 failed: {e}")

    # Method 6: Grandparent or nearby elements
    try:
        methods_tried.append("grandparent_search")
        grandparent = element.evaluate_handle("el => el.parentElement?.parentElement")
        if grandparent:
            # Look for any text content in grandparent that looks like a question
            gp_text = grandparent.inner_text().strip()
            if (gp_text and len(gp_text) > 5 and len(gp_text) < 200 and
                ('?' in gp_text or
                 any(word in gp_text.lower() for word in ['do you', 'are you', 'have you', 'what', 'how', 'when', 'where', 'why']))):
                question_text = gp_text
                confidence = 0.4
                logger.info(f"Method 6 (grandparent_search) found: '{question_text}' (confidence: {confidence})")
                return question_text, confidence, "grandparent_search"
    except Exception as e:
        logger.debug(f"Method 6 failed: {e}")

    logger.info(f"All methods tried for element: {methods_tried}")
    return None, 0.0, "none"


def detect_questions_with_ai(page, logger):
    """
    Use AI to detect questions that traditional methods might have missed.
    This is a fallback method when no questions are found.
    """
    logger.info("Attempting AI-assisted question detection...")

    try:
        # Get page content
        page_content = page.content()

        # Create AI prompt for question detection
        ai_prompt = f"""You are analyzing a job application webpage to identify form questions that need to be answered.

Page URL: {page.url}
Page Content (HTML snippet):
{page_content[:3000]}{"..." if len(page_content) > 3000 else ""}

Please identify all the questions or form fields that require user input on this job application page. Look for:

1. Text input fields with labels
2. Textarea fields for longer answers
3. Dropdown/select menus
4. Radio button groups
5. Checkbox questions
6. Any other form elements that appear to be questions

For each question you find, provide:
- The question text or label
- The type of input (text, textarea, select, radio, checkbox)
- A confidence score (0.0 to 1.0) of how certain you are this is actually a question

Format your response as:
QUESTION: [question text or label]
TYPE: [field type]
CONFIDENCE: [confidence score]

Example:
QUESTION: What is your full name?
TYPE: text
CONFIDENCE: 0.9

QUESTION: Please select your experience level
TYPE: select
CONFIDENCE: 0.8

Return only the questions you find, one per block. If you find no questions, return "NO_QUESTIONS_FOUND"."""

        # Call AI (using existing openrouter config if available)
        try:
            from config import config
            openrouter_config = config.get("question_answering", {}).get("openrouter", {})
        except:
            openrouter_config = {"enabled": False}

        if not openrouter_config.get("enabled", False):
            logger.info("AI detection disabled - skipping")
            return []

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_config.get("api_key", ""),
        )

        response = client.chat.completions.create(
            model=openrouter_config.get("model", "anthropic/claude-3.5-sonnet"),
            messages=[{"role": "user", "content": ai_prompt}],
            timeout=30
        )

        ai_response = response.choices[0].message.content.strip()
        logger.info(f"AI question detection response: {ai_response}")

        # Parse AI response
        questions = []
        blocks = ai_response.split('\n\n')

        for block in blocks:
            if 'QUESTION:' in block and 'TYPE:' in block and 'CONFIDENCE:' in block:
                try:
                    question_line = next(line for line in block.split('\n') if line.startswith('QUESTION:'))
                    type_line = next(line for line in block.split('\n') if line.startswith('TYPE:'))
                    confidence_line = next(line for line in block.split('\n') if line.startswith('CONFIDENCE:'))

                    question_text = question_line.replace('QUESTION:', '').strip()
                    field_type = type_line.replace('TYPE:', '').strip().lower()
                    confidence = float(confidence_line.replace('CONFIDENCE:', '').strip())

                    if confidence > 0.5:  # Only include reasonably confident detections
                        # Create a dummy element for AI-detected questions
                        # In a real implementation, we'd need to find the actual element
                        # For now, we'll skip these since we can't fill them without the element
                        logger.info(f"AI-detected question: '{question_text}' (type: {field_type}, confidence: {confidence})")
                        # Note: We can't actually fill these without the DOM element reference
                        # This is more for logging and future enhancement

                except Exception as e:
                    logger.debug(f"Error parsing AI question block: {e}")

        logger.info(f"AI-assisted detection completed")
        return questions

    except Exception as e:
        logger.error(f"Error in AI-assisted question detection: {e}")
        return []


def debug_question_detection(page, logger):
    """
    Debug function to help troubleshoot question detection issues.
    This will log detailed information about all form elements found.
    """
    logger.info("=== DEBUG QUESTION DETECTION ===")

    try:
        # Get all form elements
        all_inputs = page.query_selector_all('input, textarea, select')
        logger.info(f"Found {len(all_inputs)} total form elements")

        for i, elem in enumerate(all_inputs, 1):
            tag_name = elem.evaluate("el => el.tagName").lower()
            elem_type = elem.get_attribute('type') or 'text'
            elem_id = elem.get_attribute('id') or 'no-id'
            elem_name = elem.get_attribute('name') or 'no-name'
            placeholder = elem.get_attribute('placeholder') or 'no-placeholder'
            aria_label = elem.get_attribute('aria-label') or 'no-aria-label'

            logger.info(f"Element {i}: {tag_name}[type={elem_type}] id='{elem_id}' name='{elem_name}'")
            logger.info(f"  Placeholder: '{placeholder}'")
            logger.info(f"  Aria-label: '{aria_label}'")
            logger.info(f"  Visible: {elem.is_visible()}")

            # Try to find associated label
            if elem_id:
                label = page.query_selector(f'label[for="{elem_id}"]')
                if label:
                    logger.info(f"  Label found: '{label.inner_text().strip()}'")

            # Check parent for labels
            try:
                parent = elem.evaluate_handle("el => el.parentElement")
                if parent:
                    parent_labels = parent.query_selector_all("label")
                    for label in parent_labels:
                        logger.info(f"  Parent label: '{label.inner_text().strip()}'")
            except:
                pass

            # Check for preceding text
            try:
                prev_sibling = elem.evaluate_handle("el => el.previousElementSibling")
                if prev_sibling:
                    prev_text = prev_sibling.inner_text().strip()
                    if prev_text and len(prev_text) > 3:
                        logger.info(f"  Previous sibling text: '{prev_text[:50]}...'")
            except:
                pass

        logger.info("=== DEBUG DETECTION COMPLETE ===")
        return len(all_inputs)

    except Exception as e:
        logger.error(f"Error in debug question detection: {e}")
        return 0


def answer_question(page, element, question_text, field_type, bank, logger, progress, qa_mode, qa_on_unknown, qa_fuzzy_threshold, resume_context, openrouter_config):
    """
    Main function to answer a question using hybrid approach.
    Returns True if answered successfully, False otherwise.
    """
    logger.info(f"=== STARTING QUESTION ANSWER PROCESS ===")
    logger.info(f"Question: '{question_text}'")
    logger.info(f"Field type: {field_type}")
    logger.info(f"QA Mode: {qa_mode}")

    answer = None
    answer_source = None

    # Step 1: Try to find answer in question bank (if mode allows)
    logger.info("STEP 1: Attempting to find answer in question bank")
    if qa_mode in ["stored_only", "hybrid"]:
        logger.info("Searching question bank for matches...")
        matched_q, score = find_matching_question(bank, question_text, qa_fuzzy_threshold)
        if matched_q:
            answer = matched_q["answer"]
            answer_source = "bank"
            matched_q["match_count"] = matched_q.get("match_count", 0) + 1
            matched_q["last_used"] = datetime.now().isoformat()
            print(f"   ✅ Found answer in bank (score: {score:.1f}): {answer}")
            logger.info(f"✅ BANK MATCH FOUND - Score: {score:.1f}, Answer: '{answer}'")

            # Update stats
            bank["stats"]["total_answered_from_bank"] = bank["stats"].get("total_answered_from_bank", 0) + 1
            if "question_stats" not in progress:
                progress["question_stats"] = {"total_answered": 0, "from_bank": 0, "from_ai": 0, "skipped": 0}
            progress["question_stats"]["from_bank"] += 1
            progress["question_stats"]["total_answered"] += 1
        else:
            logger.info(f"No match found in question bank (threshold: {qa_fuzzy_threshold})")

    # Step 2: If no match and mode allows AI, try AI
    if not answer and qa_mode in ["hybrid", "ai_only"]:
        print(f"   🔍 No match found in question bank, calling AI...")
        logger.info("STEP 2: No bank match found, attempting AI answer")
        print(f"   🤖 Calling AI for: {question_text}")
        logger.info(f"Calling AI for question: '{question_text}'")

        ai_answer = ask_ai_for_answer(question_text, resume_context, logger, openrouter_config)

        if ai_answer:
            answer = ai_answer
            answer_source = "ai"
            print(f"   ✅ AI answered: {answer}")
            print(f"   💾 Saving AI answer to question bank for future use")
            logger.info(f"✅ AI ANSWER RECEIVED: '{answer}'")

            # Save AI answer to bank for future use
            add_question_to_bank(bank, question_text, answer, source="ai")
            save_question_bank(bank)

            # Update stats
            bank["stats"]["total_answered_from_ai"] = bank["stats"].get("total_answered_from_ai", 0) + 1
            if "question_stats" not in progress:
                progress["question_stats"] = {"total_answered": 0, "from_bank": 0, "from_ai": 0, "skipped": 0}
            progress["question_stats"]["from_ai"] += 1
            progress["question_stats"]["total_answered"] += 1

            logger.info(f"AI answer saved to question bank")
        else:
            print(f"   ❌ AI failed to provide answer for: {question_text}")
            logger.warning(f"❌ AI failed to provide answer for: '{question_text}'")

    # Step 3: If still no answer, try alternative approaches
    if not answer:
        print(f"   ⚠️ No answer found for: {question_text}")
        logger.warning(f"No answer found using bank or standard AI for: '{question_text}'")

        # Try 1: Enhanced AI prompting with more context
        print(f"   🔄 Trying enhanced AI prompting...")
        logger.info("STEP 3A: Attempting enhanced AI prompting")
        enhanced_ai_answer = ask_ai_for_answer_enhanced(question_text, resume_context, logger, openrouter_config)

        if enhanced_ai_answer:
            answer = enhanced_ai_answer
            answer_source = "ai_enhanced"
            print(f"   ✅ Enhanced AI answered: {answer}")
            logger.info(f"✅ ENHANCED AI ANSWER: '{answer}'")

            # Save enhanced AI answer to bank
            add_question_to_bank(bank, question_text, answer, source="ai")
            save_question_bank(bank)
        else:
            print(f"   ❌ Enhanced AI also failed")
            logger.warning("Enhanced AI also failed to provide answer")

            # Try 2: Manual input (if enabled)
            if qa_on_unknown == "pause_and_beep":
                logger.info("STEP 3B: No AI answer found, requesting manual input")
                print(f"   ⌨️ Manual input required for: {question_text}")
                manual_answer = pause_for_manual_input(question_text)

                if manual_answer:
                    answer = manual_answer
                    answer_source = "manual"
                    print(f"   ✅ Manual answer provided: {answer}")
                    logger.info(f"✅ MANUAL ANSWER RECEIVED: '{answer}'")

                    # Save manual answer to bank
                    add_question_to_bank(bank, question_text, answer, source="manual")
                    save_question_bank(bank)
                else:
                    print(f"   ❌ User chose to skip this question")
                    logger.warning("User chose to skip this question")

                    # Don't return False here - try to continue with the application
                    # Only skip if this is a critical question that blocks progress
                    if is_critical_question(question_text):
                        print(f"   🚫 Critical question not answered, skipping application")
                        logger.error(f"CRITICAL QUESTION SKIPPED: '{question_text}'")
                        return False
                    else:
                        print(f"   ⚠️ Non-critical question skipped, continuing application")
                        logger.warning(f"Non-critical question skipped, continuing: '{question_text}'")
                        return True  # Continue with application even if this question isn't answered
            else:
                # Try 3: Skip this specific question but continue with application
                print(f"   ⏭️ Skipping unanswered question, continuing application")
                logger.warning(f"STEP 3C: Unable to answer question, continuing anyway: '{question_text}'")

                # Update stats
                if "question_stats" not in progress:
                    progress["question_stats"] = {"total_answered": 0, "from_bank": 0, "from_ai": 0, "skipped": 0}
                progress["question_stats"]["skipped"] += 1

                # Don't return False - try to continue with the application
                # Only skip if this is a critical question that blocks progress
                if is_critical_question(question_text):
                    print(f"   🚫 Critical question not answered, skipping application")
                    logger.error(f"CRITICAL QUESTION SKIPPED: '{question_text}'")
                    return False
                else:
                    print(f"   ⚠️ Non-critical question skipped, continuing application")
                    logger.warning(f"Non-critical question skipped, continuing: '{question_text}'")
                    return True  # Continue with application even if this question isn't answered

    # Step 4: Fill in the answer based on field type
    logger.info("STEP 4: Filling form field with answer")
    logger.info(f"Answer to use: '{answer}' (source: {answer_source})")
    try:
        if field_type == "text" or field_type == "textarea":
            logger.info(f"Filling {field_type} field with answer")

            # Check if this is a custom form element (div that looks like input)
            tag_name = element.evaluate("el => el.tagName").lower()
            if tag_name == "div":
                logger.info("📋 Detected custom div-based form field, trying alternative filling methods")
                success = fill_custom_form_element(element, answer, logger)
                if success:
                    logger.info(f"✅ Successfully filled custom {field_type} field")
                    print(f"   📝 Filled custom {field_type} field")
                else:
                    logger.warning(f"❌ Could not fill custom {field_type} field")
                    return False
            else:
                # Standard form element
                element.fill(answer)
                time.sleep(0.3)  # Brief wait for value to register
                
                # Verify the fill was successful
                try:
                    filled_value = element.input_value()
                    if filled_value != answer:
                        logger.warning(f"Fill verification failed. Expected: '{answer}', Got: '{filled_value}'")
                        # Retry once
                        element.fill('')
                        time.sleep(0.2)
                        element.fill(answer)
                        time.sleep(0.3)
                        
                        # Verify again
                        filled_value = element.input_value()
                        if filled_value == answer:
                            logger.info(f"✅ Retry successful - field now contains correct value")
                        else:
                            logger.error(f"❌ Retry failed - field still incorrect")
                except Exception as e:
                    logger.debug(f"Could not verify fill: {e}")
                
                logger.info(f"✅ Successfully filled {field_type} field")
                print(f"   📝 Filled {field_type} field")

        elif field_type == "select":
            logger.info("Filling dropdown/select field")

            # Check if this is a custom form element (div that looks like select)
            tag_name = element.evaluate("el => el.tagName").lower()
            if tag_name == "div":
                logger.info("📋 Detected custom div-based select field, trying alternative filling methods")
                success = fill_custom_select_element(element, answer, logger)
                if success:
                    logger.info(f"✅ Successfully filled custom select field")
                    print(f"   📝 Filled custom select field")
                else:
                    logger.warning(f"❌ Could not fill custom select field")
                    return False
            else:
                # Standard select element
                # Try to select option that best matches the answer
                options = element.query_selector_all("option")
                logger.info(f"Found {len(options)} dropdown options")
                best_match = None
                best_score = 0

                for option in options:
                    option_text = option.inner_text().strip()
                    score = fuzz.ratio(answer.lower(), option_text.lower())
                    logger.info(f"Option: '{option_text}' - Score: {score}")
                    if score > best_score:
                        best_score = score
                        best_match = option

                if best_match and best_score > 50:
                    value = best_match.get_attribute("value")
                    option_text = best_match.inner_text().strip()
                    element.select_option(value=value)
                    logger.info(f"✅ Selected dropdown option: '{option_text}' (score: {best_score})")
                    print(f"   📝 Selected dropdown: {option_text}")
                else:
                    logger.warning(f"❌ Could not find matching option for: '{answer}' (best score: {best_score})")
                    print(f"   ❌ Could not find matching dropdown option")
                    return False

        elif field_type == "radio":
            logger.info("Filling radio button field")
            # Find all radio buttons with same name
            name = element.get_attribute("name")
            radios = page.query_selector_all(f'input[type="radio"][name="{name}"]')
            logger.info(f"Found {len(radios)} radio buttons for name: {name}")

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
                    logger.info(f"Radio option: '{label_text}' - Score: {score}")
                    if score > best_score:
                        best_score = score
                        best_match = radio

            if best_match and best_score > 50:
                best_match.check()
                logger.info(f"✅ Selected radio button (score: {best_score})")
                print(f"   📝 Selected radio button")
            else:
                logger.warning(f"❌ Could not find matching radio option for: '{answer}' (best score: {best_score})")
                print(f"   ❌ Could not find matching radio option")
                return False

        time.sleep(0.5)  # Small delay after filling
        logger.info("✅ Question answered successfully")
        logger.info("=== QUESTION ANSWER PROCESS COMPLETED ===")
        return True

    except Exception as e:
        logger.error(f"❌ Error filling answer: {e}")
        logger.error(f"Exception details: {str(e)}")
        print(f"   ❌ Error filling field: {e}")
        return False


# ============================================================================
# Application Functions (extracted from original indeed_bot.py)
# ============================================================================

def human_delay(base_delay, variation_percent=None):
    """Add human-like random variation to delays."""
    if variation_percent is None:
        variation_percent = base_delays["human_like_variation"]

    variation = base_delay * variation_percent
    delay = base_delay + random.uniform(-variation, variation)
    time.sleep(max(0.5, delay))


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


def wait_for_confirmation(page, logger, job_url, max_wait=10):
    """
    Wait for confirmation page to load after clicking submit.
    Returns True if confirmation detected, False otherwise.
    """
    print(f"   ⏳ Waiting up to {max_wait}s for confirmation page...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            current_url = page.url

            # Check URL for confirmation keywords
            url_lower = current_url.lower()
            if any(keyword in url_lower for keyword in [
                "confirmation", "submitted", "thank", "complete", "success",
                "application", "applied", "candidature", "bewerbung"
            ]):
                print(f"   ✅ Confirmation detected in URL: {current_url}")
                return True

            # Check page content for confirmation text
            if detect_confirmation_in_content(page):
                print(f"   ✅ Confirmation detected in page content")
                return True

            # Wait a bit before checking again
            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"Error while waiting for confirmation: {e}")
            time.sleep(0.5)

    print(f"   ⏰ Confirmation wait timed out after {max_wait}s")
    return False


def detect_confirmation_in_content(page):
    """
    Detect confirmation text in the current page content.
    Returns True if confirmation found, False otherwise.
    """
    try:
        # Get page content
        page_content = page.content().lower()

        # Look for confirmation keywords in content
        confirmation_keywords = [
            "application submitted", "candidature soumise", "bewerbung eingereicht",
            "application has been submitted", "candidature a été soumise",
            "your application has been", "votre candidature a été",
            "thank you for applying", "merci d'avoir postulé",
            "successfully submitted", "soumis avec succès",
            "application complete", "candidature complète",
            "we have received your application", "nous avons reçu votre candidature",
            "application sent", "candidature envoyée",
            "you have successfully applied", "vous avez postulé avec succès"
        ]

        for keyword in confirmation_keywords:
            if keyword in page_content:
                print(f"   📋 Found confirmation text: '{keyword}'")
                return True

        # Also check for specific HTML elements that indicate success
        try:
            # Look for success messages or confirmation divs
            success_elements = page.query_selector_all('[class*="success"], [class*="confirmation"], [class*="submitted"], [data-testid*="success"]')
            if success_elements:
                for elem in success_elements:
                    text = (elem.inner_text() or "").lower()
                    if any(word in text for word in ["success", "submitted", "complete", "thank"]):
                        print(f"   ✅ Found success element: '{text[:50]}...'")
                        return True
        except:
            pass

        return False

    except Exception as e:
        print(f"   ❌ Error detecting confirmation in content: {e}")
        return False


def find_continue_button(page):
    """
    Find continue/next buttons as fallback after submission.
    Returns button element or None.
    """
    try:
        # Look for continue/next buttons
        continue_selectors = [
            'button:visible:has-text("continue")',
            'button:visible:has-text("continuer")',
            'button:visible:has-text("weiter")',
            'button:visible:has-text("siguiente")',
            'button:visible:has-text("next")',
            'button:visible:has-text("suivant")',
            'button:visible:has-text("nächste")',
            'button:visible:has-text("siguiente")'
        ]

        for selector in continue_selectors:
            try:
                if ":has-text" in selector:
                    text = selector.split(":has-text(")[1].rstrip(")").strip('"')
                    btn = page.query_selector(f'button:visible:has-text("{text}")')
                else:
                    btn = page.query_selector(selector)

                if btn and btn.is_visible():
                    btn_text = (btn.inner_text() or "").lower()
                    if not any(skip in btn_text for skip in ["cancel", "close", "back"]):
                        print(f"   🔄 Found continue button: '{btn_text}'")
                        return btn
            except:
                continue

        return None

    except Exception as e:
        print(f"   ❌ Error finding continue button: {e}")
        return None


def apply_to_job(browser, job_url, language, logger, progress, question_bank, qa_enabled, qa_mode, qa_on_unknown, qa_fuzzy_threshold, resume_context, openrouter_config, max_retries=3):
    """Open a new tab, apply to the job, log the result, and close the tab."""
    # Extract job ID for cleaner logging
    job_id = job_url.split('jk=')[-1].split('&')[0] if 'jk=' in job_url else job_url.split('/')[-1]

    for attempt in range(max_retries):
        page = browser.new_page()
        try:
            page.goto(job_url)
            page.wait_for_load_state("domcontentloaded")
            human_delay(base_delays["page_load"])
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
                click_and_wait(apply_btn, 5)
            else:
                logger.warning(
                    f"No Indeed Apply button found for {job_url}")
                page.close()
                return False

            # add timeout for the wizard loop
            start_time = time.time()
            while True:
                if time.time() - start_time > 40:
                    logger.warning(
                        f"Timeout applying to {job_url}, closing tab and moving to next.")
                    break
                current_url = page.url
                # Resume step: select resume card if present
                print(f"📄 Looking for resume card...")
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

                    # Look for continue button after resume selection
                    print(f"🔄 Looking for continue button after resume...")
                    continuer_btn = None
                    btns = page.query_selector_all('button:visible')
                    for btn in btns:
                        text = (btn.inner_text() or "").lower()
                        if "continuer" in text or "continue" in text:
                            continuer_btn = btn
                            break
                    if continuer_btn:
                        print(f"✅ Found continue button, clicking...")
                        click_and_wait(continuer_btn, 3)
                        print(f"⏭️  Continuing to next step...")
                        continue  # go to next step
                    else:
                        print(f"⚠️ No continue button found after resume selection")
                else:
                    print(f"📋 No resume card found, continuing...")

                # QUESTION ANSWERING: Detect and answer any questions on this page
                if qa_enabled:
                    try:
                        print(f"🔍 Scanning for questions on application page...")
                        logger.info("Starting question detection and answering process")

                        detected_questions = detect_question_fields(page)
                        logger.info(f"Initial question detection found {len(detected_questions)} questions")

                        # Log details about detected questions for debugging
                        if detected_questions:
                            for i, (elem, q_text, f_type, conf, method) in enumerate(detected_questions[:3], 1):  # Show first 3
                                logger.info(f"  Question {i}: '{q_text}' (type: {f_type}, confidence: {conf:.2f}, method: {method})")
                            if len(detected_questions) > 3:
                                logger.info(f"  ... and {len(detected_questions) - 3} more questions")
                        else:
                            logger.warning("No questions detected - enhanced detection may have failed")

                        # If no questions detected, try again with a delay (dynamic content)
                        if not detected_questions:
                            print(f"   ⏳ No questions found, waiting for dynamic content...")
                            time.sleep(2)
                            detected_questions = detect_question_fields(page)
                            logger.info(f"Second attempt found {len(detected_questions)} questions after waiting")

                            # Log details about second attempt
                            if detected_questions:
                                logger.info("✅ Second attempt found questions - dynamic content loaded successfully")
                                for i, (elem, q_text, f_type, conf, method) in enumerate(detected_questions[:3], 1):
                                    logger.info(f"  Question {i}: '{q_text}' (type: {f_type}, confidence: {conf:.2f}, method: {method})")
                            else:
                                logger.warning("❌ Second attempt also found no questions - enhanced detection failed")

                        if detected_questions:
                            print(f"📝 Found {len(detected_questions)} question(s) to answer")
                            logger.info(f"=== QUESTION DETECTION SUMMARY ===")
                            for i, (elem, q_text, f_type, conf, method) in enumerate(detected_questions, 1):
                                logger.info(f"Q{i}: '{q_text}' (type: {f_type}, confidence: {conf:.2f}, method: {method})")
                                print(f"   {i}. {q_text}")
                                print(f"      Field type: {f_type}, Confidence: {conf:.2f}, Method: {method}")

                            questions_answered = 0
                            questions_skipped = 0
                            critical_questions_unanswered = []

                            for i, (elem, q_text, f_type, conf, method) in enumerate(detected_questions, 1):
                                logger.info(f"Processing question {i}/{len(detected_questions)}: '{q_text}'")

                                # Verify field value and re-attempt filling if needed (double-check approach)
                                try:
                                    should_skip = False
                                    if f_type in ["text", "textarea"]:
                                        current_value = elem.input_value()
                                        if current_value and len(current_value.strip()) > 0:
                                            # Field has content - but is it the RIGHT content?
                                            # Only skip if the value looks intentional (not placeholder text)
                                            if not looks_like_placeholder(current_value):
                                                logger.info(f"Field already filled with valid content: '{current_value[:50]}'")
                                                should_skip = True
                                            else:
                                                logger.info(f"Field contains placeholder text: '{current_value[:50]}' - will attempt to fill")
                                    
                                    if should_skip:
                                        print(f"   ✓ Field already filled, skipping")
                                        logger.info(f"Skipping already filled field: {q_text}")
                                        continue
                                except Exception as e:
                                    logger.debug(f"Error checking field value: {e}")
                                    # If we can't check, attempt to fill anyway
                                    pass

                                # Try to answer the question with enhanced persistence
                                print(f"   🔄 Attempting to answer question {i}...")
                                logger.info(f"Attempting to answer: '{q_text}' (type: {f_type}, confidence: {conf:.2f})")

                                answered = answer_question(page, elem, q_text, f_type, question_bank, logger, progress, qa_mode, qa_on_unknown, qa_fuzzy_threshold, resume_context, openrouter_config)

                                if answered:
                                    print(f"   ✅ Question {i} answered successfully")
                                    logger.info(f"✅ Question {i} answered successfully: '{q_text}'")
                                    questions_answered += 1
                                else:
                                    print(f"   ❌ Question {i} could not be answered")
                                    logger.warning(f"❌ Question {i} could not be answered: '{q_text}'")
                                    questions_skipped += 1

                                    # Track critical questions that couldn't be answered
                                    if is_critical_question(q_text):
                                        critical_questions_unanswered.append(q_text)
                                        logger.error(f"CRITICAL QUESTION UNANSWERED: '{q_text}'")

                            # Save question bank after answering questions
                            if detected_questions:
                                save_question_bank(question_bank)
                                save_progress(progress)
                                print(f"💾 Question bank updated with {len(detected_questions)} questions")
                                logger.info(f"Saved {len(detected_questions)} questions to question bank")

                            # Show detailed summary of question answering results
                            logger.info(f"=== QUESTION ANSWERING SUMMARY ===")
                            logger.info(f"Total questions: {len(detected_questions)}")
                            logger.info(f"Questions answered: {questions_answered}")
                            logger.info(f"Questions skipped: {questions_skipped}")
                            logger.info(f"Critical questions unanswered: {len(critical_questions_unanswered)}")

                            print(f"📊 Question answering summary: {questions_answered} answered, {questions_skipped} skipped")

                            if questions_answered > 0:
                                print(f"✅ Application can continue with {questions_answered} questions answered")
                            logger.info(f"✅ Application can continue with {questions_answered} questions answered")
                            if questions_skipped > 0:
                                print(f"⚠️ {questions_skipped} questions were skipped but application will continue")
                            logger.warning(f"⚠️ {questions_skipped} questions were skipped but application will continue")

                            # CRITICAL: If there are unanswered critical questions, don't proceed
                            if critical_questions_unanswered:
                                print(f"🚫 CRITICAL: {len(critical_questions_unanswered)} critical questions could not be answered")
                                logger.error(f"CRITICAL: {len(critical_questions_unanswered)} critical questions unanswered: {critical_questions_unanswered}")
                                logger.error("Cannot proceed with application due to unanswered critical questions")
                                page.close()
                                return False

                            # AI VERIFICATION: Before proceeding, use AI to check if there are still unanswered questions
                            print(f"🔍 AI verification: Checking if all questions are properly answered...")
                            logger.info("Performing AI verification to check for unanswered questions")

                            verification_result = verify_all_questions_answered(page, logger, openrouter_config, resume_context)
                            if verification_result:
                                print(f"✅ AI verification passed - all questions appear to be answered")
                                logger.info("✅ AI verification passed - all questions appear to be answered")
                            else:
                                print(f"⚠️ AI verification found potential unanswered questions")
                                logger.warning("⚠️ AI verification found potential unanswered questions")
                                # For now, continue anyway but log the warning
                                # In the future, we might want to make this stricter

                        else:
                            print(f"📋 No questions detected on this page")
                            logger.info("No questions detected on this page")

                    except Exception as e:
                        print(f"❌ Error in question answering: {e}")
                        logger.error(f"Error in question answering: {e}")
                        logger.error(f"Exception details: {str(e)}")

                # Continue with application flow after question answering
                print(f"🔄 Continuing application flow after question answering...")

                # Try to find a submit button with comprehensive strategies
                submit_btn = None
                print(f"🔍 Looking for submit/continue button...")

                # First, try specific submit selectors
                submit_selectors = [
                    # Language-specific submit texts (most specific first)
                    'button:visible:has-text("déposer ma candidature")',
                    'button:visible:has-text("soumettre")',
                    'button:visible:has-text("submit application")',
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
                    'button:visible:has-text("next")',
                    'button:visible:has-text("finish")',
                    'button:visible:has-text("complete")',
                    'button:visible:has-text("send")',
                    # Type-based selectors (most reliable)
                    'button[type="submit"]:visible',
                    'input[type="submit"]:visible'
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
                            btn_text = (submit_btn.inner_text() or "").lower()
                            # Verify it's not a cancel/close button
                            if not any(skip in btn_text for skip in ["cancel", "close", "annuler", "fermer", "schließen"]):
                                print(f"✅ Found submit button: '{btn_text}'")
                                break
                    except Exception as e:
                        continue

                # Enhanced fallback: look through all visible buttons for submit-like text
                if not submit_btn:
                    print(f"🔄 Submit button not found with selectors, trying fallback...")
                    try:
                        all_buttons = page.query_selector_all('button:visible, input[type="submit"]:visible')
                        best_match = None
                        best_score = 0

                        for btn in all_buttons:
                            try:
                                btn_text = (btn.inner_text() or "").lower()
                                aria_label = (btn.get_attribute("aria-label") or "").lower()
                                combined_text = f"{btn_text} {aria_label}"

                                # Skip unwanted buttons
                                if any(skip in combined_text for skip in ["cancel", "close", "annuler", "fermer", "schließen", "back", "previous"]):
                                    continue

                                # Look for submit-related keywords
                                submit_keywords = [
                                    "submit", "apply", "send", "finish", "complete", "done",
                                    "soumettre", "postuler", "terminer", "finaliser",
                                    "bewerben", "senden", "absenden", "postular",
                                    "enviar", "finalizar", "invia", "solliciteren",
                                    "ansøg", "continue", "continuer", "weiter",
                                    "siguiente", "next"
                                ]

                                score = 0
                                for keyword in submit_keywords:
                                    if keyword in combined_text:
                                        score += 1

                                # Prefer submit buttons over continue buttons
                                if "submit" in combined_text or "soumettre" in combined_text or "bewerben" in combined_text:
                                    score += 2

                                if score > best_score:
                                    best_score = score
                                    best_match = btn
                                    print(f"   📊 Found potential submit button: '{btn_text}' (score: {score})")

                            except Exception as e:
                                continue

                        if best_match and best_score > 0:
                            submit_btn = best_match
                            btn_text = (submit_btn.inner_text() or "").lower()
                            print(f"✅ Selected best submit button: '{btn_text}' (score: {best_score})")
                        else:
                            print(f"❌ No suitable submit button found")

                    except Exception as e:
                        print(f"❌ Error in submit button fallback: {e}")

                # If we found a submit button, click it and wait for confirmation
                if submit_btn:
                    print(f"🚀 Clicking submit button...")
                    click_and_wait(submit_btn, 3)

                    # Wait for confirmation page with robust detection
                    print(f"⏳ Waiting for submission confirmation...")
                    confirmation_detected = wait_for_confirmation(page, logger, job_url)

                    if confirmation_detected:
                        print(f"🎉 Application submitted successfully!")
                        logger.info(f"Applied successfully to {job_url}")
                        break
                    else:
                        print(f"⚠️ Confirmation not detected, trying alternative approach...")
                        # Try to detect confirmation in current page content
                        if detect_confirmation_in_content(page):
                            print(f"🎉 Application submitted successfully (content confirmation)!")
                            logger.info(f"Applied successfully to {job_url}")
                            break
                        else:
                            print(f"❌ Could not confirm submission, trying continue...")
                            # Try to find and click continue button as fallback
                            continue_btn = find_continue_button(page)
                            if continue_btn:
                                print(f"🔄 Found continue button, clicking...")
                                click_and_wait(continue_btn, 2)
                                # Wait a bit and check again
                                time.sleep(3)
                                if detect_confirmation_in_content(page):
                                    print(f"🎉 Application submitted successfully!")
                                    logger.info(f"Applied successfully to {job_url}")
                                    break
                else:
                    print(f"❌ No submit button found on page")
                    logger.warning(f"No submit button found at {current_url}")

                # Final fallback: try any enabled button as last resort
                print(f"🔄 Trying final fallback button search...")
                fallback_btn = page.query_selector('button[type="button"]:not([aria-disabled="true"]):visible, button[type="submit"]:not([aria-disabled="true"]):visible')
                if fallback_btn:
                    btn_text = (fallback_btn.inner_text() or "").lower()
                    print(f"⚠️ Using fallback button: '{btn_text}'")
                    click_and_wait(fallback_btn, 2)

                    # Check for confirmation after fallback
                    time.sleep(3)
                    if detect_confirmation_in_content(page):
                        print(f"🎉 Application submitted successfully!")
                        logger.info(f"Applied successfully to {job_url}")
                        break

                print(f"❌ Could not complete application")
                logger.warning(f"Could not complete application for {job_url}")
                break
            page.close()
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {job_url}: {e}")
            page.close()
            if attempt < max_retries - 1:
                wait_time = base_delays["retry_delay"] * (attempt + 1)  # Progressive delay
                print(f"Retrying in {wait_time} seconds... (attempt {attempt + 2}/{max_retries})")
                human_delay(wait_time)
                continue
            else:
                logger.error(f"Failed to apply to {job_url} after {max_retries} attempts: {e}")
                return False


def setup_logger():
    logger = logging.getLogger("indeed_apply")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/indeed_bot.log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def main():
    """Main application function."""
    global qa_config, qa_enabled, qa_mode, qa_on_unknown, qa_fuzzy_threshold, openrouter_config, resume_context, base_delays

    config = load_config()
    logger = setup_logger()
    progress = load_progress()
    question_bank = load_question_bank()

    # Get configuration
    qa_config = config.get("question_answering", {})
    qa_enabled = qa_config.get("enabled", False)
    qa_mode = qa_config.get("mode", "hybrid")
    qa_on_unknown = qa_config.get("on_unknown_question", "ai_then_skip")
    qa_fuzzy_threshold = qa_config.get("fuzzy_match_threshold", 80)
    openrouter_config = qa_config.get("openrouter", {})
    resume_context = qa_config.get("resume_context", "")

    timing_config = config.get("timing", {})
    base_delays = {
        "page_load": timing_config.get("page_load", 3),
        "between_applications": timing_config.get("between_applications", 5),
        "element_wait": timing_config.get("element_wait", 2),
        "retry_delay": timing_config.get("retry_delay", 5),
        "human_like_variation": timing_config.get("human_like_variation", 0.3)
    }

    camoufox_config = config.get("camoufox", {})
    user_data_dir = camoufox_config.get("user_data_dir")
    language = camoufox_config.get("language", "www")

    # Initialize question stats in progress if not present
    if "question_stats" not in progress:
        progress["question_stats"] = {
            "total_answered": 0,
            "from_bank": 0,
            "from_ai": 0,
            "skipped": 0
        }

    # Load jobs from CSV
    jobs = load_jobs_from_csv()
    if not jobs:
        print("❌ No jobs to apply to. Run indeed_scrape.py first.")
        return

    print(f"🚀 Starting Indeed job applier...")
    print(f"📋 Will apply to {len(jobs)} jobs")

    with Camoufox(user_data_dir=user_data_dir, persistent_context=True) as browser:
        logger = setup_logger()

        # Check for login
        page = browser.new_page()
        page.goto("https://" + language + ".indeed.com")

        cookies = page.context.cookies()
        ppid_cookie = next(
            (cookie for cookie in cookies if cookie['name'] == 'PPID'), None)
        if not ppid_cookie:
            print("❌ Token not found, please log in to Indeed first.")
            print("🔐 Redirecting to login page...")
            print("🔄 You need to restart the applier after logging in.")
            page.goto("https://secure.indeed.com/auth?hl=" + language)
            time.sleep(1000)  # wait for manual login
            return
        else:
            print("✅ Token found, proceeding with job applications...")
            page.close()

            # Apply to each job
            for i, job_url in enumerate(jobs, 1):
                print(f"\n🔄 Applying to job {i}/{len(jobs)}: {job_url[:80]}...")
                success = apply_to_job(browser, job_url, language, logger, progress, question_bank, qa_enabled, qa_mode, qa_on_unknown, qa_fuzzy_threshold, resume_context, openrouter_config)
                update_progress(progress, job_url, success)
                save_progress(progress)

                if success:
                    print(f"✅ Successfully applied to job {i}")
                else:
                    print(f"❌ Failed to apply to job {i}")

                human_delay(base_delays["between_applications"])

            # Save final question bank state
            save_question_bank(question_bank)

            print_progress_stats(progress, len(jobs))


if __name__ == "__main__":
    main()
