"""
Test suite for the Automated Invoice Compliance Tracker

This module contains comprehensive tests to verify system functionality.
Run this after installation to ensure everything is working properly.
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import datetime, timedelta
import sqlite3

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestInvoiceTracker(unittest.TestCase):
    """Test suite for the invoice tracking system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = tempfile.mkdtemp()
        self.sample_invoice_data = {
            'invoice_id': 'TEST-001',
            'vendor_name': 'ABC Pvt Ltd',
            'total_amount': 25000.0,
            'currency': 'INR',
            'po_number': 'PO-TEST-001',
            'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'invoice_date': datetime.now().strftime('%Y-%m-%d'),
            'line_items': [
                {'description': 'Test Item', 'quantity': 1, 'rate': 25000, 'amount': 25000}
            ],
            'tax_amount': 4500.0,
            'net_amount': 20500.0
        }
    
    def test_directory_structure(self):
        """Test that all required directories exist"""
        required_dirs = [
            'incoming_invoices',
            'processed_invoices',
            'data',
            'logs',
            'exports'
        ]
        
        for directory in required_dirs:
            self.assertTrue(
                os.path.exists(directory),
                f"Required directory {directory} does not exist"
            )
    
    def test_environment_file(self):
        """Test that environment file exists"""
        self.assertTrue(
            os.path.exists('.env') or os.path.exists('.env.example'),
            "Environment file (.env or .env.example) not found"
        )
    
    def test_requirements_file(self):
        """Test that requirements file exists and contains required packages"""
        self.assertTrue(os.path.exists('requirements.txt'), "requirements.txt not found")
        
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'streamlit',
            'pandas',
            'watchdog',
            'chromadb',
            'openai',
            'PyPDF2',
            'loguru'
        ]
        
        for package in required_packages:
            self.assertIn(
                package,
                requirements,
                f"Required package {package} not found in requirements.txt"
            )
    
    def test_backend_import(self):
        """Test that backend module can be imported"""
        try:
            import backend
            self.assertTrue(hasattr(backend, 'InvoiceProcessor'))
            self.assertTrue(hasattr(backend, 'InvoiceData'))
            self.assertTrue(hasattr(backend, 'ComplianceResult'))
        except ImportError as e:
            self.fail(f"Failed to import backend module: {e}")
    
    def test_file_watcher_import(self):
        """Test that file watcher module can be imported"""
        try:
            import file_watcher
            self.assertTrue(hasattr(file_watcher, 'InvoiceWatcher'))
            self.assertTrue(hasattr(file_watcher, 'InvoiceFileHandler'))
        except ImportError as e:
            self.fail(f"Failed to import file_watcher module: {e}")
    
    def test_utils_import(self):
        """Test that utils module can be imported"""
        try:
            import utils
            self.assertTrue(hasattr(utils, 'generate_sample_invoice_pdf'))
            self.assertTrue(hasattr(utils, 'create_sample_invoices'))
        except ImportError as e:
            self.fail(f"Failed to import utils module: {e}")
    
    def test_compliance_rules(self):
        """Test compliance rule logic"""
        try:
            from backend import InvoiceProcessor, InvoiceData
            
            processor = InvoiceProcessor()
            
            # Test approved invoice
            approved_invoice = InvoiceData(
                invoice_id='TEST-APPROVED',
                vendor_name='ABC Pvt Ltd',  # Approved vendor
                total_amount=30000.0,       # Within limit
                currency='INR',
                po_number='PO-123',         # Has PO number
                due_date='2024-12-31',
                invoice_date='2024-01-01',
                line_items=[],
                tax_amount=5400.0,
                net_amount=24600.0,
                file_path='/test/path',
                extracted_text='Test invoice text'
            )
            
            result = processor.check_compliance(approved_invoice)
            self.assertTrue(result.is_approved, "Should be approved")
            self.assertEqual(result.status, 'approved')
            
            # Test flagged invoice (over amount limit)
            flagged_invoice = InvoiceData(
                invoice_id='TEST-FLAGGED',
                vendor_name='ABC Pvt Ltd',
                total_amount=75000.0,       # Over limit
                currency='INR',
                po_number='PO-123',
                due_date='2024-12-31',
                invoice_date='2024-01-01',
                line_items=[],
                tax_amount=13500.0,
                net_amount=61500.0,
                file_path='/test/path',
                extracted_text='Test invoice text'
            )
            
            result = processor.check_compliance(flagged_invoice)
            self.assertFalse(result.is_approved, "Should be flagged")
            self.assertIn('exceeds limit', str(result.reasons))
            
        except Exception as e:
            self.fail(f"Compliance test failed: {e}")
    
    def test_database_operations(self):
        """Test database creation and basic operations"""
        try:
            from backend import InvoiceProcessor
            
            processor = InvoiceProcessor()
            
            # Test database connection
            self.assertIsNotNone(processor.conn, "Database connection should exist")
            
            # Test table creation
            cursor = processor.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn('invoices', tables, "invoices table should exist")
            self.assertIn('compliance_results', tables, "compliance_results table should exist")
            
        except Exception as e:
            self.fail(f"Database test failed: {e}")

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_sample_data_generation(self):
        """Test sample data generation"""
        try:
            from utils import create_sample_invoices
            
            # This should run without errors
            sample_invoices = create_sample_invoices()
            self.assertIsInstance(sample_invoices, list)
            self.assertGreater(len(sample_invoices), 0)
            
            # Check if PDF files were created
            invoice_files = [
                f for f in os.listdir('./incoming_invoices') 
                if f.endswith('.pdf')
            ]
            self.assertGreater(len(invoice_files), 0, "No PDF files were generated")
            
        except Exception as e:
            self.fail(f"Sample data generation failed: {e}")
    
    def test_streamlit_app_structure(self):
        """Test that the Streamlit app has required components"""
        try:
            with open('app.py', 'r') as f:
                app_content = f.read()
            
            # Check for required Streamlit components
            required_components = [
                'st.set_page_config',
                'st.session_state',
                'st.tabs',
                'st.dataframe',
                'st.metric'
            ]
            
            for component in required_components:
                self.assertIn(
                    component,
                    app_content,
                    f"Required Streamlit component {component} not found"
                )
            
        except Exception as e:
            self.fail(f"Streamlit app structure test failed: {e}")

def run_system_health_check():
    """Run a comprehensive system health check"""
    print("🏥 Running System Health Check...")
    print("=" * 50)
    
    checks = [
        ("📁 Directory Structure", check_directories),
        ("📄 Required Files", check_files),
        ("🐍 Python Modules", check_imports),
        ("🗄️ Database", check_database),
        ("⚙️ Configuration", check_configuration),
        ("📊 Sample Data", check_sample_data)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if result:
                print(f"✅ {check_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {check_name}: FAILED")
        except Exception as e:
            print(f"❌ {check_name}: ERROR - {e}")
    
    print("=" * 50)
    print(f"📊 Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 System is healthy and ready to run!")
        print("\n🚀 To start the application:")
        print("   python run.py")
        print("   # or")
        print("   streamlit run app.py")
    else:
        print("⚠️ Some issues found. Please fix them before running the application.")
    
    return passed == total

def check_directories():
    """Check if all required directories exist"""
    required_dirs = [
        'incoming_invoices', 'processed_invoices', 'data', 'logs', 'exports'
    ]
    return all(os.path.exists(d) for d in required_dirs)

def check_files():
    """Check if all required files exist"""
    required_files = [
        'app.py', 'backend.py', 'file_watcher.py', 'utils.py', 
        'requirements.txt', 'README.md'
    ]
    return all(os.path.exists(f) for f in required_files)

def check_imports():
    """Check if core modules can be imported"""
    try:
        import backend
        import file_watcher
        import utils
        return True
    except ImportError:
        return False

def check_database():
    """Check database functionality"""
    try:
        from backend import InvoiceProcessor
        processor = InvoiceProcessor()
        return processor.conn is not None
    except:
        return False

def check_configuration():
    """Check configuration files"""
    return os.path.exists('.env') or os.path.exists('.env.example')

def check_sample_data():
    """Check if sample data can be generated"""
    try:
        from utils import create_sample_invoices
        return True
    except:
        return False

if __name__ == "__main__":
    # Run health check
    run_system_health_check()
    
    print("\n🧪 Running Unit Tests...")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
