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
    Detect all question fields on the current page.
    Returns list of tuples: (element, question_text, field_type)
    """
    questions = []

    try:
        # Find all input fields, textareas, and select elements
        input_elements = page.query_selector_all('input[type="text"], input:not([type]), textarea')
        select_elements = page.query_selector_all('select')
        radio_groups = {}

        # Process text inputs and textareas
        for elem in input_elements:
            if not elem.is_visible():
                continue

            # Try to find associated label or question text
            question_text = None

            # Method 1: Check for associated label (most reliable)
            elem_id = elem.get_attribute("id")
            if elem_id:
                label = page.query_selector(f'label[for="{elem_id}"]')
                if label:
                    question_text = label.inner_text().strip()

            # Method 2: Check for nearby label in parent
            if not question_text:
                try:
                    parent = elem.evaluate_handle("el => el.parentElement")
                    label = parent.query_selector("label")
                    if label:
                        question_text = label.inner_text().strip()
                except:
                    pass

            # Method 3: Check for preceding text (question above the field)
            if not question_text:
                try:
                    # Look for text before the field that looks like a question
                    prev_sibling = elem.evaluate_handle("el => el.previousElementSibling")
                    if prev_sibling:
                        prev_text = prev_sibling.inner_text().strip()
                        # Check if it looks like a question (ends with ? or contains question words)
                        if (prev_text and len(prev_text) > 5 and
                            ('?' in prev_text or
                             any(word in prev_text.lower() for word in ['do you', 'are you', 'have you', 'what', 'how', 'when', 'where', 'why']))):
                            question_text = prev_text
                except:
                    pass

            # Method 4: Check placeholder (fallback)
            if not question_text:
                placeholder = elem.get_attribute("placeholder")
                if placeholder and len(placeholder) > 3:
                    question_text = placeholder

            # Method 5: Check aria-label (fallback)
            if not question_text:
                aria_label = elem.get_attribute("aria-label")
                if aria_label:
                    question_text = aria_label

            # Filter out very short or option-like text
            if question_text and len(question_text) > 3:
                # Skip if it looks like a form option (single words, numbers, etc.)
                if not (len(question_text.split()) <= 3 and
                       question_text.lower() not in ['yes', 'no'] and
                       not any(char.isdigit() for char in question_text)):
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

            # Check for preceding question text
            if not question_text:
                try:
                    prev_sibling = elem.evaluate_handle("el => el.previousElementSibling")
                    if prev_sibling:
                        prev_text = prev_sibling.inner_text().strip()
                        if (prev_text and len(prev_text) > 5 and
                            ('?' in prev_text or
                             any(word in prev_text.lower() for word in ['do you', 'are you', 'have you', 'what', 'how', 'when', 'where', 'why']))):
                            question_text = prev_text
                except:
                    pass

            if not question_text:
                aria_label = elem.get_attribute("aria-label")
                if aria_label:
                    question_text = aria_label

            if question_text and len(question_text) > 3:
                questions.append((elem, question_text, "select"))

        # Process radio buttons (group by name) - be more selective
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

                # Check for preceding question text
                if not question_text:
                    try:
                        prev_sibling = elem.evaluate_handle("el => el.previousElementSibling")
                        if prev_sibling:
                            prev_text = prev_sibling.inner_text().strip()
                            if (prev_text and len(prev_text) > 5 and
                                ('?' in prev_text or
                                 any(word in prev_text.lower() for word in ['do you', 'are you', 'have you', 'what', 'how', 'when', 'where', 'why']))):
                                question_text = prev_text
                    except:
                        pass

                if question_text and len(question_text) > 3:
                    radio_groups[name] = (elem, question_text, "radio")

        questions.extend(radio_groups.values())

    except Exception as e:
        print(f"Error detecting question fields: {e}")

    return questions


def answer_question(page, element, question_text, field_type, bank, logger, progress, qa_mode, qa_on_unknown, qa_fuzzy_threshold, resume_context, openrouter_config):
    """
    Main function to answer a question using hybrid approach.
    Returns True if answered successfully, False otherwise.
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
            print(f"   ✅ Found answer in bank (score: {score:.1f}): {answer}")
            logger.info(f"Found match in bank (score: {score:.1f}): {answer}")

            # Update stats
            bank["stats"]["total_answered_from_bank"] = bank["stats"].get("total_answered_from_bank", 0) + 1
            if "question_stats" not in progress:
                progress["question_stats"] = {"total_answered": 0, "from_bank": 0, "from_ai": 0, "skipped": 0}
            progress["question_stats"]["from_bank"] += 1
            progress["question_stats"]["total_answered"] += 1

    # Step 2: If no match and mode allows AI, try AI
    if not answer and qa_mode in ["hybrid", "ai_only"]:
        print(f"   🔍 No match found in question bank, calling AI...")
        logger.info("No match in bank, trying AI...")
        print(f"   🤖 Calling AI for: {question_text}")
        ai_answer = ask_ai_for_answer(question_text, resume_context, logger, openrouter_config)

        if ai_answer:
            answer = ai_answer
            answer_source = "ai"
            print(f"   ✅ AI answered: {answer}")
            print(f"   💾 Saving AI answer to question bank for future use")

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
        else:
            print(f"   ❌ AI failed to provide answer for: {question_text}")
            logger.warning(f"AI failed to provide answer for: {question_text}")

    # Step 3: If still no answer, try alternative approaches
    if not answer:
        print(f"   ⚠️ No answer found for: {question_text}")

        # Try 1: Enhanced AI prompting with more context
        print(f"   🔄 Trying enhanced AI prompting...")
        enhanced_ai_answer = ask_ai_for_answer_enhanced(question_text, resume_context, logger, openrouter_config)

        if enhanced_ai_answer:
            answer = enhanced_ai_answer
            answer_source = "ai_enhanced"
            print(f"   ✅ Enhanced AI answered: {answer}")

            # Save enhanced AI answer to bank
            add_question_to_bank(bank, question_text, answer, source="ai")
            save_question_bank(bank)
        else:
            print(f"   ❌ Enhanced AI also failed")

            # Try 2: Manual input (if enabled)
            if qa_on_unknown == "pause_and_beep":
                logger.info("No answer found, pausing for manual input...")
                print(f"   ⌨️ Manual input required for: {question_text}")
                manual_answer = pause_for_manual_input(question_text)

                if manual_answer:
                    answer = manual_answer
                    answer_source = "manual"
                    print(f"   ✅ Manual answer provided: {answer}")

                    # Save manual answer to bank
                    add_question_to_bank(bank, question_text, answer, source="manual")
                    save_question_bank(bank)
                else:
                    print(f"   ❌ User chose to skip this question")
                    logger.warning("User skipped this question")

                    # Don't return False here - try to continue with the application
                    # Only skip if this is a critical question that blocks progress
                    if is_critical_question(question_text):
                        print(f"   🚫 Critical question not answered, skipping application")
                        return False
                    else:
                        print(f"   ⚠️ Non-critical question skipped, continuing application")
                        return True  # Continue with application even if this question isn't answered
            else:
                # Try 3: Skip this specific question but continue with application
                print(f"   ⏭️ Skipping unanswered question, continuing application")
                logger.warning(f"Unable to answer question but continuing: {question_text}")

                # Update stats
                if "question_stats" not in progress:
                    progress["question_stats"] = {"total_answered": 0, "from_bank": 0, "from_ai": 0, "skipped": 0}
                progress["question_stats"]["skipped"] += 1

                # Don't return False - try to continue with the application
                # Only skip if this is a critical question that blocks progress
                if is_critical_question(question_text):
                    print(f"   🚫 Critical question not answered, skipping application")
                    return False
                else:
                    print(f"   ⚠️ Non-critical question skipped, continuing application")
                    return True  # Continue with application even if this question isn't answered

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
        return True

    except Exception as e:
        logger.error(f"Error filling answer: {e}")
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
                        detected_questions = detect_question_fields(page)

                        # If no questions detected, try again with a delay (dynamic content)
                        if not detected_questions:
                            print(f"   ⏳ No questions found, waiting for dynamic content...")
                            time.sleep(2)
                            detected_questions = detect_question_fields(page)

                        if detected_questions:
                            print(f"📝 Found {len(detected_questions)} question(s) to answer")
                            logger.info(f"Detected {len(detected_questions)} question(s) on page")

                            questions_answered = 0
                            questions_skipped = 0

                            for i, (elem, q_text, f_type) in enumerate(detected_questions, 1):
                                print(f"   {i}. {q_text}")
                                print(f"      Field type: {f_type}")

                                # Skip if field already has a value
                                try:
                                    current_value = elem.input_value() if f_type in ["text", "textarea"] else None
                                    if current_value and len(current_value.strip()) > 0:
                                        print(f"   ✓ Field already filled, skipping")
                                        logger.info(f"Skipping already filled field: {q_text}")
                                        continue
                                except:
                                    pass

                                # Try to answer the question with enhanced persistence
                                print(f"   🔄 Attempting to answer question {i}...")
                                answered = answer_question(page, elem, q_text, f_type, question_bank, logger, progress, qa_mode, qa_on_unknown, qa_fuzzy_threshold, resume_context, openrouter_config)

                                if answered:
                                    print(f"   ✅ Question {i} answered successfully")
                                    questions_answered += 1
                                else:
                                    print(f"   ❌ Question {i} could not be answered")
                                    questions_skipped += 1

                            # Save question bank after answering questions
                            if detected_questions:
                                save_question_bank(question_bank)
                                save_progress(progress)
                                print(f"💾 Question bank updated with {len(detected_questions)} questions")

                            # Show summary of question answering results
                            if questions_answered > 0 or questions_skipped > 0:
                                print(f"📊 Question answering summary: {questions_answered} answered, {questions_skipped} skipped")
                                if questions_answered > 0:
                                    print(f"✅ Application can continue with {questions_answered} questions answered")
                                if questions_skipped > 0:
                                    print(f"⚠️ {questions_skipped} questions were skipped but application will continue")
                        else:
                            print(f"📋 No questions detected on this page")

                    except Exception as e:
                        print(f"❌ Error in question answering: {e}")
                        logger.error(f"Error in question answering: {e}")

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
