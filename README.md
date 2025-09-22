# 📋 Automated Invoice Compliance Tracker

A comprehensive **AI-powered invoice processing and compliance system** built with Python, Streamlit, and modern ML technologies. This hackathon-ready project automatically processes PDF invoices, checks compliance rules, detects anomalies, and provides an interactive dashboard for monitoring and querying invoice data.

## 🌟 Features

### Core Functionality
- **📄 Automatic PDF Processing**: Monitors `incoming_invoices/` folder using watchdog
- **🤖 AI-Powered Data Extraction**: Uses OpenAI GPT to extract structured data from invoices
- **✅ Smart Compliance Checking**: Auto-approves invoices based on configurable business rules
- **🚩 Intelligent Flagging**: Flags non-compliant invoices with detailed reasons
- **📊 Vector Database Storage**: Stores invoice chunks in ChromaDB for advanced RAG queries

### Advanced Features
- **🔍 Anomaly Detection**: ML-based detection using Isolation Forest
- **🛡️ Fraud Detection**: Multi-signal fraud detection (text patterns, amounts, vendor analysis)
- **📧 Email Notifications**: Automatic alerts for flagged invoices
- **💬 AI Chat Interface**: Natural language querying of invoice data
- **📈 Real-time Dashboard**: Live metrics, visualizations, and analytics
- **📤 Export & Reporting**: CSV, Excel, JSON exports with filtering

### Technical Highlights
- **🎛️ Session State Management**: Real-time updates using Streamlit session state
- **🔄 Background Processing**: Concurrent file monitoring and processing
- **📚 RAG System**: Retrieval-Augmented Generation for intelligent Q&A
- **🎨 Modern UI**: Clean, responsive Streamlit interface with custom CSS
- **🛠️ Modular Architecture**: Well-structured, maintainable codebase

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Tesseract OCR (for image processing)

### Installation

1. **Clone and navigate to the project:**
```bash
git clone <repository-url>
cd automated_invoice_tracker
```

2. **Create virtual environment (recommended):**
```bash
python -m venv invoice_tracker_env
source invoice_tracker_env/bin/activate  # On Linux/Mac
# or
invoice_tracker_env\Scripts\activate     # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Start the application:**
```bash
python run.py
```

**Or manually:**
```bash
streamlit run app.py
```

Access the dashboard at: http://localhost:8501

## 📁 Project Structure

```
automated_invoice_tracker/
├── app.py                  # Main Streamlit dashboard
├── backend.py              # Core processing engine
├── file_watcher.py         # File monitoring service
├── utils.py                # Utilities and sample data generation
├── run.py                  # Startup script
├── requirements.txt        # Python dependencies
├── .env                    # Environment configuration
├── README.md              # This file
│
├── incoming_invoices/      # Drop new PDF invoices here
├── processed_invoices/     # Processed files organized by status
│   ├── approved/
│   ├── flagged/
│   ├── pending/
│   └── error/
│
├── data/                   # Data storage
│   ├── invoices.db        # SQLite database
│   ├── chroma_db/         # Vector database
│   └── config.json        # System configuration
│
├── logs/                   # Application logs
├── exports/               # Generated reports and exports
└── sample_invoices/       # Sample test data
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Compliance Rules
MAX_INVOICE_AMOUNT=50000
APPROVED_VENDORS=ABC Pvt Ltd,XYZ Logistics

# Email Notifications
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_TO=admin@company.com

# System Configuration
LOG_LEVEL=INFO
CHROMA_DB_PATH=./data/chroma_db
```

### Compliance Rules

The system auto-approves invoices when ALL conditions are met:
- ✅ Amount ≤ ₹50,000 (configurable)
- ✅ Vendor in approved list: ["ABC Pvt Ltd", "XYZ Logistics"]
- ✅ PO number exists
- ✅ Valid invoice data

Invoices are flagged when any rule fails, with detailed reasons provided.

## 🎯 Usage Guide

### 1. Processing Invoices

**Automatic Processing:**
- Drop PDF invoices in `incoming_invoices/` folder
- System automatically detects, processes, and moves files
- Real-time status updates in dashboard

**Manual Upload:**
- Use the "Settings" tab in the dashboard
- Upload multiple PDFs simultaneously
- Monitor processing progress

### 2. Dashboard Navigation

**📈 Dashboard Tab:**
- Key metrics (total, approved, flagged invoices)
- Interactive visualizations and analytics
- Processing timeline and vendor analysis

**📄 Invoice Details Tab:**
- Detailed invoice table with filtering
- Individual invoice drill-down
- Compliance reasons and scores

**💬 AI Chat Tab:**
- Natural language queries about invoices
- Examples: "Why was INV-001 flagged?", "Show ABC Pvt Ltd invoices"
- Context-aware responses using RAG

**⚙️ Settings Tab:**
- Manual file upload
- System configuration
- Database management

**📤 Export Tab:**
- Generate reports (CSV, Excel, JSON)
- Filter by status, date range
- Compliance and vendor analysis

### 3. Sample Queries

The AI chat interface supports various queries:

```
"Why was invoice INV-2024001 flagged?"
"Show me all invoices from ABC Pvt Ltd"
"How many invoices were processed today?"
"What's the total amount of approved invoices?"
"Which vendors have the most flagged invoices?"
"Show me invoices over ₹30,000"
```

## 🔧 Advanced Features

### Anomaly Detection
- Uses Isolation Forest ML algorithm
- Analyzes invoice amounts, vendor patterns, text features
- Configurable sensitivity threshold
- Automatic flagging of suspicious invoices

### Fraud Detection
- Multi-signal approach combining:
  - Suspicious keyword detection
  - Round number analysis
  - Vendor name validation
  - Missing information patterns
- Real-time scoring and alerts

### Email Notifications
- Automatic alerts for flagged invoices
- Configurable SMTP settings
- Rich email templates with invoice details
- Attachment support for reports

### Vector Database (ChromaDB)
- Stores invoice text chunks with metadata
- Enables semantic search and RAG queries
- Persistent storage across sessions
- Scalable for large datasets

## 🛠️ Development

### Adding New Compliance Rules

Edit `backend.py` in the `check_compliance` method:

```python
def check_compliance(self, invoice_data: InvoiceData) -> ComplianceResult:
    # Add your custom rules here
    if custom_condition:
        reasons.append("Custom rule failed")
        is_approved = False
```

### Extending the Dashboard

Add new tabs or metrics in `app.py`:

```python
def render_custom_tab():
    st.subheader("Custom Analysis")
    # Your custom visualizations here

# Add to main tabs
tabs = st.tabs(["Dashboard", "Details", "Chat", "Settings", "Export", "Custom"])
with tabs[5]:
    render_custom_tab()
```

### Custom ML Models

Extend the anomaly/fraud detection in `backend.py`:

```python
def custom_detector(self, invoice_data: InvoiceData) -> float:
    # Implement your custom detection logic
    return score
```

## 📊 Data Schema

### Invoice Data Structure
```python
@dataclass
class InvoiceData:
    invoice_id: str
    vendor_name: str
    total_amount: float
    currency: str
    po_number: Optional[str]
    due_date: Optional[str]
    invoice_date: Optional[str]
    line_items: List[Dict]
    tax_amount: Optional[float]
    net_amount: Optional[float]
    file_path: str
    extracted_text: str
```

### Database Tables
- **invoices**: Core invoice data
- **compliance_results**: Processing results and scores
- **ChromaDB**: Vector embeddings for RAG

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors:**
```bash
# Install missing dependencies
pip install -r requirements.txt

# For OCR functionality
sudo apt-get install tesseract-ocr  # Linux
brew install tesseract              # Mac
```

**2. API Key Issues:**
```bash
# Check .env file
cat .env | grep OPENAI_API_KEY

# Test API connection
python -c "import openai; openai.api_key='your_key'; print('OK')"
```

**3. File Processing Issues:**
- Check file permissions in `incoming_invoices/`
- Verify PDF files are not corrupted
- Check logs in `logs/invoice_tracker.log`

**4. Database Issues:**
```bash
# Reset database
rm data/invoices.db
rm -rf data/chroma_db/
python utils.py  # Reinitialize
```

### Performance Optimization

**For Large Datasets:**
- Increase ChromaDB chunk size
- Configure batch processing
- Use database indexing
- Implement pagination in UI

**For High Throughput:**
- Scale file watcher threads
- Use async processing
- Implement queue management
- Add load balancing

## 🧪 Testing

### Generate Sample Data
```bash
python utils.py
```

### Run System Tests
```bash
python -c "from utils import run_system_tests; run_system_tests()"
```

### Manual Testing
1. Place sample PDFs in `incoming_invoices/`
2. Monitor processing in dashboard
3. Test compliance rules with different amounts/vendors
4. Verify email notifications
5. Test chat interface queries

## 🚀 Deployment

### Local Production
```bash
# Use production-grade WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8501 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Use Procfile with web process
- **AWS/GCP**: Use container services
- **Azure**: App Service with Python runtime

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Hackathon Ready

This project is designed to be **hackathon-ready** with:

- ✅ **Complete functionality** out of the box
- ✅ **Modern tech stack** (AI, ML, Vector DB)
- ✅ **Professional UI/UX** with Streamlit
- ✅ **Comprehensive documentation**
- ✅ **Sample data** for immediate testing
- ✅ **Modular architecture** for easy extension
- ✅ **Production considerations** (logging, error handling)

## 📞 Support

For questions or issues:
- Check the troubleshooting section above
- Review logs in `logs/invoice_tracker.log`
- Open an issue on GitHub
- Contact the development team

---

**Built with ❤️ for efficient invoice processing and compliance automation**
