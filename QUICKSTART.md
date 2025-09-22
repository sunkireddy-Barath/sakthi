# 🚀 Quick Start Guide - Automated Invoice Compliance Tracker

## 30-Second Setup

```bash
# 1. Navigate to project directory
cd automated_invoice_tracker

# 2. Run setup script
./setup.sh

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key to .env file
echo "OPENAI_API_KEY=your_key_here" >> .env

# 5. Start the application
python run.py
```

**🌐 Open http://localhost:8501 in your browser**

## ⚡ Instant Demo

1. **The app starts with sample invoices already generated**
2. **Dashboard shows real-time metrics and visualizations**
3. **Try the AI chat: "Why was invoice INV-001 flagged?"**
4. **Upload new PDFs in the Settings tab**

## 🎯 Key Features to Showcase

### 📊 Dashboard
- **Live metrics** (total, approved, flagged invoices)
- **Interactive visualizations** (status overview, timeline)
- **Vendor analysis** with amounts and compliance scores

### 🤖 AI-Powered Processing
- **Drop PDFs** in `incoming_invoices/` folder → **Auto-processed**
- **Smart extraction** using OpenAI GPT
- **Compliance rules**: Amount ≤ ₹50K, Approved vendors, PO required

### 💬 Intelligent Chat
Ask natural language questions:
- "Show me all ABC Pvt Ltd invoices"
- "Why are invoices getting flagged?"
- "What's the total approved amount?"

### 🔍 Advanced Analytics
- **Anomaly detection** using ML algorithms
- **Fraud detection** with multiple signals
- **Email alerts** for flagged invoices

## 🛠️ Customization

### Change Compliance Rules
```python
# Edit backend.py
MAX_AMOUNT = 100000  # Change limit
APPROVED_VENDORS = ["Your", "Vendors", "Here"]
```

### Add Custom Analytics
```python
# Edit app.py - add new tabs or metrics
def render_custom_analysis():
    st.subheader("Custom Analysis")
    # Your code here
```

## 🚨 Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**No API key?**
```bash
# Edit .env file
OPENAI_API_KEY=sk-your_actual_key_here
```

**Can't process PDFs?**
```bash
# Install Tesseract OCR
sudo apt-get install tesseract-ocr  # Linux
brew install tesseract              # Mac
```

## 📱 Demo Script

1. **Show Dashboard** - "Here's our real-time invoice monitoring system"
2. **Upload Invoice** - "Watch it get processed automatically"
3. **Show Compliance** - "It checks business rules and flags violations"
4. **Use AI Chat** - "Ask questions in natural language"
5. **Show Analytics** - "ML-powered anomaly and fraud detection"
6. **Export Data** - "Generate reports in multiple formats"

## 🏆 Hackathon Highlights

- ✅ **Full-stack solution** (Backend + Frontend + ML)
- ✅ **Modern tech stack** (Streamlit, ChromaDB, OpenAI)
- ✅ **Real-world applicability** (Actual business problem)
- ✅ **Scalable architecture** (Modular, extensible)
- ✅ **Production-ready** (Logging, error handling, tests)

---

**💡 Pro Tip**: The system works best with real-looking invoice PDFs. The sample generator creates realistic invoices for demo purposes!
