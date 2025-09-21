"""
Automated Invoice Compliance Tracker - Backend Module

This module handles all the core invoice processing functionality including:
- PDF text extraction and OCR
- LLM-based invoice data extraction
- Compliance rule checking
- ChromaDB vector storage
- Anomaly detection
- Email notifications
- Export functionality

Author: AI Assistant
Date: September 2025
"""

import os
import re
import json
import logging
import smtplib
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import email.mime.text
import email.mime.multipart  
import email.mime.base
import email.encoders

# PDF Processing
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image

# OpenCV with graceful fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenCV not available - {e}")
    CV2_AVAILABLE = False
    cv2 = None

# Vector Database with graceful fallback
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    print("✅ ChromaDB imported successfully")
except Exception as e:
    print(f"⚠️ ChromaDB not available: {e}")
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None

# LLM Integration with graceful fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    print("✅ OpenAI imported successfully")
except Exception as e:
    print(f"⚠️ OpenAI not available: {e}")
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain imported successfully")
except Exception as e:
    print(f"⚠️ LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None
    Document = None
from langchain.schema import Document

# Machine Learning
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# Plotting (configure backend for headless deployment)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Environment
from dotenv import load_dotenv

# Logging
from loguru import logger

# Load environment variables
load_dotenv()

@dataclass
class InvoiceData:
    """Data structure for storing extracted invoice information"""
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
    
@dataclass
class ComplianceResult:
    """Data structure for compliance check results"""
    invoice_id: str
    is_approved: bool
    status: str  # 'approved', 'flagged', 'pending'
    reasons: List[str]
    compliance_score: float
    processed_at: datetime
    anomaly_score: Optional[float] = None
    fraud_score: Optional[float] = None

class InvoiceProcessor:
    """Main class for processing invoices and managing compliance"""
    
    def __init__(self):
        """Initialize the invoice processor with configuration and models"""
        self.setup_logging()
        self.load_config()
        self.setup_database()
        self.setup_chromadb()
        self.setup_ml_models()
        
        logger.info("Invoice Processor initialized successfully")
    
    def setup_logging(self):
        """Configure logging for the application"""
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_file = os.getenv('LOG_FILE', './logs/invoice_tracker.log')
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logger.add(
            log_file,
            rotation="10 MB",
            retention="30 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
        )
    
    def load_config(self):
        """Load configuration from environment variables"""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.max_amount = float(os.getenv('MAX_INVOICE_AMOUNT', 50000))
        self.approved_vendors = [v.strip() for v in os.getenv('APPROVED_VENDORS', '').split(',')]
        self.chroma_db_path = os.getenv('CHROMA_DB_PATH', './data/chroma_db')
        self.collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'invoice_embeddings')
        
        # Email configuration
        self.email_config = {
            'smtp_server': os.getenv('EMAIL_SMTP_SERVER'),
            'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', 587)),
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'from_email': os.getenv('EMAIL_FROM'),
            'to_email': os.getenv('EMAIL_TO')
        }
        
        # Set OpenAI API key and initialize client
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
    
    def setup_database(self):
        """Initialize SQLite database for storing invoice records"""
        db_path = './data/invoices.db'
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create tables
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT UNIQUE,
                vendor_name TEXT,
                total_amount REAL,
                currency TEXT,
                po_number TEXT,
                due_date TEXT,
                invoice_date TEXT,
                line_items TEXT,
                tax_amount REAL,
                net_amount REAL,
                file_path TEXT,
                extracted_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS compliance_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT,
                is_approved BOOLEAN,
                status TEXT,
                reasons TEXT,
                compliance_score REAL,
                anomaly_score REAL,
                fraud_score REAL,
                processed_at TIMESTAMP,
                FOREIGN KEY (invoice_id) REFERENCES invoices (invoice_id)
            )
        ''')
        
        self.conn.commit()
        logger.info("Database initialized successfully")
    
    def setup_chromadb(self):
        """Initialize ChromaDB for vector storage and RAG"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - RAG features will be disabled")
            self.chroma_client = None
            self.collection = None
            return
            
        try:
            # Ensure ChromaDB directory exists
            os.makedirs(self.chroma_db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Invoice text embeddings for RAG system"}
                )
            
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
            self.collection = None
    
    def setup_ml_models(self):
        """Initialize machine learning models for anomaly and fraud detection"""
        # Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=float(os.getenv('ANOMALY_DETECTION_THRESHOLD', 0.05)),
            random_state=42
        )
        
        # Scaler for preprocessing
        self.scaler = StandardScaler()
        
        # Initialize with some dummy data to avoid fitting issues
        dummy_data = np.random.rand(100, 5)  # 5 features
        self.anomaly_detector.fit(dummy_data)
        self.scaler.fit(dummy_data)
        
        logger.info("ML models initialized successfully")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF using multiple methods
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text = ""
            
            # Method 1: PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                logger.info(f"Successfully extracted text using PyPDF2 from {file_path}")
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
            
            # Method 2: pdfplumber (if PyPDF2 didn't work well)
            if len(text.strip()) < 100:  # If extraction was poor
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    logger.info(f"Successfully extracted text using pdfplumber from {file_path}")
                except Exception as e:
                    logger.warning(f"pdfplumber extraction failed: {e}")
            
            # Method 3: OCR with Tesseract (if other methods failed)
            if len(text.strip()) < 50:  # If extraction was still poor
                try:
                    # Convert PDF to images and OCR
                    import pdf2image
                    images = pdf2image.convert_from_path(file_path)
                    for image in images:
                        ocr_text = pytesseract.image_to_string(image)
                        text += ocr_text + "\n"
                    logger.info(f"Successfully extracted text using OCR from {file_path}")
                except Exception as e:
                    logger.error(f"OCR extraction failed: {e}")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"All text extraction methods failed for {file_path}: {e}")
            return ""
    
    def extract_invoice_data_with_llm(self, text: str, file_path: str) -> InvoiceData:
        """
        Extract structured invoice data using LLM
        
        Args:
            text: Raw text extracted from PDF
            file_path: Path to the original PDF file
            
        Returns:
            Structured invoice data
        """
        try:
            # Create a comprehensive prompt for invoice data extraction
            prompt = f"""
            Extract the following information from this invoice text. Return the data in JSON format:
            
            Required fields:
            - invoice_id: The invoice number/ID
            - vendor_name: Company name that issued the invoice
            - total_amount: Total amount (numeric value only, no currency symbols)
            - currency: Currency code (e.g., INR, USD, EUR)
            - po_number: Purchase Order number (if present)
            - due_date: Payment due date (YYYY-MM-DD format)
            - invoice_date: Invoice date (YYYY-MM-DD format)
            - tax_amount: Tax amount (numeric value only)
            - net_amount: Net amount before tax (numeric value only)
            - line_items: Array of items with description, quantity, rate, amount
            
            Invoice Text:
            {text[:4000]}  # Limit text to avoid token limits
            
            Return only valid JSON without any additional text or formatting:
            """
            
            # Call OpenAI API
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")
                
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert invoice data extraction assistant. Extract information accurately and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the response
            extracted_data = json.loads(response.choices[0].message.content.strip())
            
            # Create InvoiceData object with validation
            invoice_data = InvoiceData(
                invoice_id=extracted_data.get('invoice_id', f"INV_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                vendor_name=extracted_data.get('vendor_name', 'Unknown Vendor'),
                total_amount=float(extracted_data.get('total_amount', 0)),
                currency=extracted_data.get('currency', 'INR'),
                po_number=extracted_data.get('po_number'),
                due_date=extracted_data.get('due_date'),
                invoice_date=extracted_data.get('invoice_date'),
                line_items=extracted_data.get('line_items', []),
                tax_amount=float(extracted_data.get('tax_amount', 0)) if extracted_data.get('tax_amount') else None,
                net_amount=float(extracted_data.get('net_amount', 0)) if extracted_data.get('net_amount') else None,
                file_path=file_path,
                extracted_text=text
            )
            
            logger.info(f"Successfully extracted invoice data for {invoice_data.invoice_id}")
            return invoice_data
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Return a fallback InvoiceData object
            return InvoiceData(
                invoice_id=f"INV_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                vendor_name="Unknown Vendor",
                total_amount=0.0,
                currency="INR",
                po_number=None,
                due_date=None,
                invoice_date=None,
                line_items=[],
                tax_amount=None,
                net_amount=None,
                file_path=file_path,
                extracted_text=text
            )
    
    def check_compliance(self, invoice_data: InvoiceData) -> ComplianceResult:
        """
        Check invoice compliance against business rules
        
        Args:
            invoice_data: Extracted invoice data
            
        Returns:
            Compliance check result
        """
        reasons = []
        is_approved = True
        compliance_score = 1.0
        
        # Rule 1: Amount check
        if invoice_data.total_amount > self.max_amount:
            reasons.append(f"Amount ₹{invoice_data.total_amount:,.2f} exceeds limit of ₹{self.max_amount:,.2f}")
            is_approved = False
            compliance_score -= 0.4
        
        # Rule 2: Vendor check
        if invoice_data.vendor_name not in self.approved_vendors:
            reasons.append(f"Vendor '{invoice_data.vendor_name}' not in approved list: {self.approved_vendors}")
            is_approved = False
            compliance_score -= 0.3
        
        # Rule 3: PO number check
        if not invoice_data.po_number:
            reasons.append("Missing PO number")
            is_approved = False
            compliance_score -= 0.2
        
        # Rule 4: Due date validation
        if invoice_data.due_date:
            try:
                due_date = datetime.strptime(invoice_data.due_date, '%Y-%m-%d')
                if due_date < datetime.now():
                    reasons.append(f"Invoice due date {invoice_data.due_date} has passed")
                    compliance_score -= 0.1
            except:
                reasons.append("Invalid due date format")
                compliance_score -= 0.1
        
        # Rule 5: Basic data validation
        if not invoice_data.invoice_id:
            reasons.append("Missing invoice ID")
            is_approved = False
            compliance_score -= 0.2
        
        if invoice_data.total_amount <= 0:
            reasons.append("Invalid total amount")
            is_approved = False
            compliance_score -= 0.3
        
        # Determine status
        if is_approved:
            status = "approved"
        elif len(reasons) > 2:
            status = "flagged"
        else:
            status = "pending"
        
        # Ensure compliance score is not negative
        compliance_score = max(0.0, compliance_score)
        
        result = ComplianceResult(
            invoice_id=invoice_data.invoice_id,
            is_approved=is_approved,
            status=status,
            reasons=reasons,
            compliance_score=compliance_score,
            processed_at=datetime.now()
        )
        
        logger.info(f"Compliance check completed for {invoice_data.invoice_id}: {status}")
        return result
    
    def detect_anomalies(self, invoice_data: InvoiceData) -> float:
        """
        Detect anomalies in invoice data using ML
        
        Args:
            invoice_data: Invoice data to analyze
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        try:
            # Create feature vector for anomaly detection
            features = [
                invoice_data.total_amount,
                len(invoice_data.vendor_name),
                len(invoice_data.line_items),
                hash(invoice_data.vendor_name) % 1000,  # Vendor hash as feature
                len(invoice_data.extracted_text)
            ]
            
            # Reshape for sklearn
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features_array)
            
            # Get anomaly score
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            
            # Convert to 0-1 scale (higher = more anomalous)
            normalized_score = 1 / (1 + np.exp(anomaly_score))  # Sigmoid transformation
            
            logger.info(f"Anomaly score for {invoice_data.invoice_id}: {normalized_score:.3f}")
            return normalized_score
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return 0.5  # Return neutral score on error
    
    def detect_fraud(self, invoice_data: InvoiceData) -> float:
        """
        Detect potential fraud in invoice using multiple signals
        
        Args:
            invoice_data: Invoice data to analyze
            
        Returns:
            Fraud score (0-1, higher = more suspicious)
        """
        try:
            fraud_indicators = []
            
            # Check for suspicious patterns in text
            text_lower = invoice_data.extracted_text.lower()
            
            # Suspicious keywords
            suspicious_keywords = ['urgent', 'immediate', 'asap', 'wire transfer', 'bitcoin', 'crypto']
            keyword_count = sum(1 for keyword in suspicious_keywords if keyword in text_lower)
            fraud_indicators.append(keyword_count / len(suspicious_keywords))
            
            # Check for round numbers (often fraudulent)
            if invoice_data.total_amount % 100 == 0 and invoice_data.total_amount > 1000:
                fraud_indicators.append(0.3)
            else:
                fraud_indicators.append(0.0)
            
            # Check vendor name consistency
            vendor_inconsistency = 0
            if len(invoice_data.vendor_name) < 3:
                vendor_inconsistency += 0.4
            if any(char.isdigit() for char in invoice_data.vendor_name):
                vendor_inconsistency += 0.2
            fraud_indicators.append(min(vendor_inconsistency, 1.0))
            
            # Check for missing critical information
            missing_info_score = 0
            if not invoice_data.po_number:
                missing_info_score += 0.2
            if not invoice_data.due_date:
                missing_info_score += 0.2
            if not invoice_data.line_items:
                missing_info_score += 0.3
            fraud_indicators.append(min(missing_info_score, 1.0))
            
            # Calculate overall fraud score
            fraud_score = np.mean(fraud_indicators)
            
            logger.info(f"Fraud score for {invoice_data.invoice_id}: {fraud_score:.3f}")
            return fraud_score
            
        except Exception as e:
            logger.error(f"Fraud detection failed: {e}")
            return 0.0
    
    def store_in_chromadb(self, invoice_data: InvoiceData, compliance_result: ComplianceResult):
        """
        Store invoice data and text chunks in ChromaDB for RAG
        
        Args:
            invoice_data: Invoice data to store
            compliance_result: Compliance check result
        """
        try:
            if not self.collection:
                logger.warning("ChromaDB collection not available")
                return
            
            # Split text into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )
            
            text_chunks = text_splitter.split_text(invoice_data.extracted_text)
            
            # Prepare documents for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(text_chunks):
                documents.append(chunk)
                metadatas.append({
                    'invoice_id': invoice_data.invoice_id,
                    'vendor_name': invoice_data.vendor_name,
                    'total_amount': invoice_data.total_amount,
                    'currency': invoice_data.currency,
                    'status': compliance_result.status,
                    'compliance_score': compliance_result.compliance_score,
                    'chunk_index': i,
                    'file_path': invoice_data.file_path,
                    'processed_at': compliance_result.processed_at.isoformat()
                })
                ids.append(f"{invoice_data.invoice_id}_chunk_{i}")
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored {len(text_chunks)} chunks for {invoice_data.invoice_id} in ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to store in ChromaDB: {e}")
    
    def save_to_database(self, invoice_data: InvoiceData, compliance_result: ComplianceResult):
        """
        Save invoice data and compliance results to SQLite database
        
        Args:
            invoice_data: Invoice data to save
            compliance_result: Compliance check result
        """
        try:
            # Insert invoice data
            self.conn.execute('''
                INSERT OR REPLACE INTO invoices 
                (invoice_id, vendor_name, total_amount, currency, po_number, due_date, 
                 invoice_date, line_items, tax_amount, net_amount, file_path, extracted_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                invoice_data.invoice_id,
                invoice_data.vendor_name,
                invoice_data.total_amount,
                invoice_data.currency,
                invoice_data.po_number,
                invoice_data.due_date,
                invoice_data.invoice_date,
                json.dumps(invoice_data.line_items),
                invoice_data.tax_amount,
                invoice_data.net_amount,
                invoice_data.file_path,
                invoice_data.extracted_text
            ))
            
            # Insert compliance result
            self.conn.execute('''
                INSERT INTO compliance_results 
                (invoice_id, is_approved, status, reasons, compliance_score, 
                 anomaly_score, fraud_score, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                compliance_result.invoice_id,
                compliance_result.is_approved,
                compliance_result.status,
                json.dumps(compliance_result.reasons),
                compliance_result.compliance_score,
                compliance_result.anomaly_score,
                compliance_result.fraud_score,
                compliance_result.processed_at
            ))
            
            self.conn.commit()
            logger.info(f"Saved {invoice_data.invoice_id} to database")
            
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
    
    def send_notification(self, invoice_data: InvoiceData, compliance_result: ComplianceResult):
        """
        Send email notification for flagged invoices
        
        Args:
            invoice_data: Invoice data
            compliance_result: Compliance check result
        """
        try:
            if compliance_result.status != 'flagged' or not self.email_config['smtp_server']:
                return
            
            # Create email content
            subject = f"Invoice Flagged: {invoice_data.invoice_id} - {invoice_data.vendor_name}"
            
            body = f"""
            Invoice Compliance Alert
            
            Invoice Details:
            - Invoice ID: {invoice_data.invoice_id}
            - Vendor: {invoice_data.vendor_name}
            - Amount: {invoice_data.currency} {invoice_data.total_amount:,.2f}
            - PO Number: {invoice_data.po_number or 'N/A'}
            - Due Date: {invoice_data.due_date or 'N/A'}
            
            Compliance Issues:
            {chr(10).join(f"- {reason}" for reason in compliance_result.reasons)}
            
            Compliance Score: {compliance_result.compliance_score:.2f}
            Anomaly Score: {compliance_result.anomaly_score:.3f if compliance_result.anomaly_score else 'N/A'}
            Fraud Score: {compliance_result.fraud_score:.3f if compliance_result.fraud_score else 'N/A'}
            
            File: {invoice_data.file_path}
            Processed: {compliance_result.processed_at}
            
            Please review this invoice manually.
            """
            
            # Send email
            msg = email.mime.multipart.MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = subject
            
            msg.attach(email.mime.text.MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_email'], text)
            server.quit()
            
            logger.info(f"Email notification sent for {invoice_data.invoice_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def process_invoice(self, file_path: str) -> Tuple[InvoiceData, ComplianceResult]:
        """
        Main function to process a single invoice through the entire pipeline
        
        Args:
            file_path: Path to the PDF invoice file
            
        Returns:
            Tuple of invoice data and compliance result
        """
        try:
            logger.info(f"Processing invoice: {file_path}")
            
            # Step 1: Extract text from PDF
            extracted_text = self.extract_text_from_pdf(file_path)
            if not extracted_text:
                raise ValueError("No text could be extracted from PDF")
            
            # Step 2: Extract structured data using LLM
            invoice_data = self.extract_invoice_data_with_llm(extracted_text, file_path)
            
            # Step 3: Check compliance rules
            compliance_result = self.check_compliance(invoice_data)
            
            # Step 4: Run anomaly detection
            anomaly_score = self.detect_anomalies(invoice_data)
            compliance_result.anomaly_score = anomaly_score
            
            # Step 5: Run fraud detection
            fraud_score = self.detect_fraud(invoice_data)
            compliance_result.fraud_score = fraud_score
            
            # Step 6: Update status based on ML scores
            if anomaly_score > 0.7 or fraud_score > 0.6:
                compliance_result.status = 'flagged'
                compliance_result.is_approved = False
                if anomaly_score > 0.7:
                    compliance_result.reasons.append(f"High anomaly score: {anomaly_score:.3f}")
                if fraud_score > 0.6:
                    compliance_result.reasons.append(f"High fraud score: {fraud_score:.3f}")
            
            # Step 7: Store in databases
            self.save_to_database(invoice_data, compliance_result)
            self.store_in_chromadb(invoice_data, compliance_result)
            
            # Step 8: Send notifications if needed
            self.send_notification(invoice_data, compliance_result)
            
            # Step 9: Move processed file
            self.move_processed_file(file_path, compliance_result.status)
            
            logger.info(f"Successfully processed {invoice_data.invoice_id} with status: {compliance_result.status}")
            return invoice_data, compliance_result
            
        except Exception as e:
            logger.error(f"Failed to process invoice {file_path}: {e}")
            # Create error result
            error_result = ComplianceResult(
                invoice_id=f"ERROR_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                is_approved=False,
                status='error',
                reasons=[f"Processing error: {str(e)}"],
                compliance_score=0.0,
                processed_at=datetime.now()
            )
            return None, error_result
    
    def move_processed_file(self, file_path: str, status: str):
        """
        Move processed file to appropriate folder
        
        Args:
            file_path: Original file path
            status: Processing status (approved/flagged/error)
        """
        try:
            import shutil
            
            filename = os.path.basename(file_path)
            processed_dir = os.path.join('./processed_invoices', status)
            os.makedirs(processed_dir, exist_ok=True)
            
            destination = os.path.join(processed_dir, filename)
            shutil.move(file_path, destination)
            
            logger.info(f"Moved {filename} to {processed_dir}")
            
        except Exception as e:
            logger.error(f"Failed to move processed file: {e}")
    
    def get_all_invoices(self) -> pd.DataFrame:
        """
        Get all invoices from database as DataFrame
        
        Returns:
            DataFrame with all invoice data
        """
        try:
            query = '''
                SELECT i.*, c.is_approved, c.status, c.reasons, c.compliance_score,
                       c.anomaly_score, c.fraud_score, c.processed_at
                FROM invoices i
                LEFT JOIN compliance_results c ON i.invoice_id = c.invoice_id
                ORDER BY i.created_at DESC
            '''
            
            df = pd.read_sql_query(query, self.conn)
            
            # Parse JSON fields
            if not df.empty:
                df['reasons'] = df['reasons'].apply(lambda x: json.loads(x) if x else [])
                df['line_items'] = df['line_items'].apply(lambda x: json.loads(x) if x else [])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get invoices: {e}")
            return pd.DataFrame()
    
    def search_invoices(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search invoices using ChromaDB RAG system
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant invoice information
        """
        try:
            if not self.collection:
                return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def export_invoices_to_csv(self, status_filter: str = None) -> str:
        """
        Export invoices to CSV file
        
        Args:
            status_filter: Filter by status (approved/flagged/pending)
            
        Returns:
            Path to exported CSV file
        """
        try:
            df = self.get_all_invoices()
            
            if status_filter:
                df = df[df['status'] == status_filter]
            
            # Prepare export directory
            export_dir = './exports'
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"invoices_{status_filter or 'all'}_{timestamp}.csv"
            file_path = os.path.join(export_dir, filename)
            
            # Export to CSV
            df.to_csv(file_path, index=False)
            
            logger.info(f"Exported {len(df)} invoices to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ""
    
    def get_dashboard_metrics(self) -> Dict:
        """
        Get metrics for dashboard display - optimized for performance
        
        Returns:
            Dictionary with various metrics
        """
        try:
            # Use direct SQL queries for better performance instead of loading full DataFrame
            cursor = self.conn.cursor()
            
            # Get basic counts
            cursor.execute("SELECT COUNT(*) FROM invoices")
            total_invoices = cursor.fetchone()[0]
            
            if total_invoices == 0:
                return {
                    'total_invoices': 0,
                    'approved_count': 0,
                    'flagged_count': 0,
                    'pending_count': 0,
                    'total_amount': 0,
                    'avg_compliance_score': 0,
                    'top_vendors': [],
                    'recent_activity': []
                }
            
            # Get status counts
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM compliance_results 
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Get total amount
            cursor.execute("SELECT SUM(total_amount) FROM invoices")
            total_amount = cursor.fetchone()[0] or 0
            
            # Get average compliance score
            cursor.execute("SELECT AVG(compliance_score) FROM compliance_results")
            avg_compliance_score = cursor.fetchone()[0] or 0
            
            # Get top vendors
            cursor.execute("""
                SELECT vendor_name, COUNT(*) as count 
                FROM invoices 
                GROUP BY vendor_name 
                ORDER BY count DESC 
                LIMIT 5
            """)
            top_vendors = dict(cursor.fetchall())
            
            metrics = {
                'total_invoices': total_invoices,
                'approved_count': status_counts.get('approved', 0),
                'flagged_count': status_counts.get('flagged', 0),
                'pending_count': status_counts.get('pending', 0),
                'total_amount': total_amount,
                'avg_compliance_score': avg_compliance_score,
                'top_vendors': top_vendors,
                'recent_activity': []  # Skip for performance - only load when needed
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    def answer_question_with_rag(self, question: str) -> str:
        """
        Answer questions about invoices using RAG system
        
        Args:
            question: User question
            
        Returns:
            AI-generated answer based on invoice data
        """
        try:
            # Search relevant invoice data
            search_results = self.search_invoices(question, top_k=3)
            
            if not search_results:
                return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about specific invoices."
            
            # Prepare context for LLM
            context = "\n\n".join([
                f"Invoice: {result['metadata']['invoice_id']}\n"
                f"Vendor: {result['metadata']['vendor_name']}\n"
                f"Amount: {result['metadata']['currency']} {result['metadata']['total_amount']}\n"
                f"Status: {result['metadata']['status']}\n"
                f"Content: {result['content']}"
                for result in search_results
            ])
            
            # Create prompt for question answering
            prompt = f"""
            Based on the following invoice information, answer the user's question:
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a helpful and accurate answer based on the invoice data:
            """
            
            # Get answer from LLM
            if not self.openai_client:
                return "OpenAI client not configured. Please check your API key."
                
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about invoice data. Be specific and cite invoice IDs when relevant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Answered question: {question}")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return f"I encountered an error while processing your question: {str(e)}"

# Global instance for use in other modules
invoice_processor = InvoiceProcessor()
