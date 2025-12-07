# 🚀 QUICKSTART - Get Running in 2 Minutes

## Option A: Automated Setup (Recommended)

```bash
# 1. Navigate to directory
cd finley-streamlit

# 2. Run setup script
bash setup.sh

# 3. Add your API key
nano .env  # or use any text editor
# Set: OPENAI_API_KEY=sk-proj-your-key-here

# 4. Run the app
streamlit run app.py
```

**Done!** App opens at http://localhost:8501

---

## Option B: Manual Setup (If script doesn't work)

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

# 3. Install packages
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here

# 5. Run app
streamlit run app.py
```

---

## 💡 First Time Using?

### 1. Open App
- Browser opens automatically to http://localhost:8501
- See the clean, minimalistic interface

### 2. Start Chatting
- Click "🎯 Get Started" for quick start
- OR type your own message

### 3. Follow Finley's Flow
- Answer questions naturally
- Finley extracts information automatically
- Watch the status bar fill up

### 4. Get Your Allocation
- After answering all questions
- Say "yes" or "show me"
- See your personalized allocation with visual chart

---

**Need help? See README.md for full documentation.**

