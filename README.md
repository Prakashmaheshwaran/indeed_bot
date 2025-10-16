# Indeed Auto-Apply Bot

**WARNING:** Use at your own risk. Indeed may change their site and break this tool.

## What It Does

- Automatically finds and applies to Indeed jobs with "Indeed Apply"
- Uses Camoufox browser automation to bypass protections
- Handles multi-step application forms and resume upload
- **NEW:** Intelligent question answering system that answers screening questions

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure `data/config.yaml`:**
   - Set your Indeed language (`fr`, `uk`, etc.)
   - Add your job search URL as `base_url`
   - Set search range (`start` and `end` - multiples of 10)

3. **Upload CV to Indeed:**
   - Go to Indeed profile and upload your resume
   - Fill in name, address, phone number

4. **Question Answering (Optional):**
   - Get free API key from [OpenRouter](https://openrouter.ai/)
   - Add to `data/config.yaml` under `question_answering`
   - Update `resume_context` with your details
   - Edit `data/questions_bank.json` with common answers

## Usage

1. **First run:** Login to Indeed manually when prompted
2. **Subsequent runs:** Bot uses saved session to apply automatically

```bash
python indeed_scrape.py  # Scrape jobs to csv/temp_jobs.csv
python indeed_apply.py   # Apply to jobs from CSV
```

## Files

- `data/config.yaml` - Bot configuration
- `data/questions_bank.json` - Stored Q&A for applications
- `data/progress.json` - Application statistics
- `logs/indeed_bot.log` - Detailed logs
- `indeed_apply.py` - Main application script
- `indeed_scrape.py` - Job scraping script

## Question Answering

- **Hybrid Mode:** Uses stored answers + AI fallback
- **Fuzzy Matching:** Finds similar questions (80% threshold)
- **Learning:** Saves AI answers for future use
- **Cost:** ~$0.001-0.01 per AI question, free after building bank

## Troubleshooting

- Check `logs/indeed_bot.log` for errors
- Ensure CV and profile info uploaded to Indeed
- Verify OpenRouter API key for question answering

## Notes

- Only works with "Indeed Apply" jobs
- May need manual captcha solving
- Use responsibly

---

**Project is for educational purposes only.**