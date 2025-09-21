"""
Utility functions and additional features for the Invoice Compliance Tracker

This module includes:
- Sample invoice generator
- Data initialization
- Advanced analytics
- Testing utilities

Author: AI Assistant
Date: September 2025
"""

import os
import json
import random
from datetime import datetime, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd

def generate_sample_invoice_pdf(filename: str, invoice_data: dict):
    """
    Generate a sample PDF invoice with given data
    
    Args:
        filename: Output PDF filename
        invoice_data: Dictionary with invoice information
    """
    try:
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Header
        title = Paragraph(f"<b>INVOICE</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Invoice details
        invoice_info = [
            ['Invoice ID:', invoice_data.get('invoice_id', 'INV-001')],
            ['Date:', invoice_data.get('invoice_date', datetime.now().strftime('%Y-%m-%d'))],
            ['Due Date:', invoice_data.get('due_date', (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'))],
            ['PO Number:', invoice_data.get('po_number', 'PO-12345')]
        ]
        
        info_table = Table(invoice_info, colWidths=[100, 200])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Vendor information
        vendor_para = Paragraph(f"<b>From:</b><br/>{invoice_data.get('vendor_name', 'ABC Pvt Ltd')}<br/>123 Business Street<br/>Mumbai, Maharashtra 400001<br/>India", styles['Normal'])
        story.append(vendor_para)
        story.append(Spacer(1, 20))
        
        # Line items
        line_items = invoice_data.get('line_items', [
            {'description': 'Product A', 'quantity': 10, 'rate': 1000, 'amount': 10000},
            {'description': 'Product B', 'quantity': 5, 'rate': 2000, 'amount': 10000}
        ])
        
        # Create line items table
        items_data = [['Description', 'Quantity', 'Rate', 'Amount']]
        for item in line_items:
            items_data.append([
                item['description'],
                str(item['quantity']),
                f"₹{item['rate']:,.2f}",
                f"₹{item['amount']:,.2f}"
            ])
        
        items_table = Table(items_data, colWidths=[200, 80, 80, 80])
        items_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(items_table)
        story.append(Spacer(1, 20))
        
        # Totals
        net_amount = invoice_data.get('net_amount', sum(item['amount'] for item in line_items))
        tax_amount = invoice_data.get('tax_amount', net_amount * 0.18)  # 18% GST
        total_amount = invoice_data.get('total_amount', net_amount + tax_amount)
        
        totals_data = [
            ['Net Amount:', f"₹{net_amount:,.2f}"],
            ['Tax (18% GST):', f"₹{tax_amount:,.2f}"],
            ['Total Amount:', f"₹{total_amount:,.2f}"]
        ]
        
        totals_table = Table(totals_data, colWidths=[200, 100])
        totals_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(totals_table)
        
        # Build PDF
        doc.build(story)
        print(f"Generated sample invoice: {filename}")
        
    except Exception as e:
        print(f"Error generating PDF: {e}")

def create_sample_invoices():
    """Create a set of sample invoices for testing"""
    
    # Ensure directories exist
    os.makedirs('./incoming_invoices', exist_ok=True)
    
    # Sample data templates
    vendors = [
        "ABC Pvt Ltd",
        "XYZ Logistics", 
        "Tech Solutions Inc",
        "Global Supplies Co",
        "Metro Services Ltd"
    ]
    
    sample_invoices = []
    
    for i in range(10):
        # Generate random invoice data
        vendor = random.choice(vendors)
        is_approved_vendor = vendor in ["ABC Pvt Ltd", "XYZ Logistics"]
        
        # Generate amount (some within limit, some over)
        if random.random() < 0.7:  # 70% chance of being within limit
            amount = random.uniform(1000, 45000)
        else:
            amount = random.uniform(55000, 150000)
        
        # Generate PO number (90% have PO numbers)
        po_number = f"PO-{random.randint(10000, 99999)}" if random.random() < 0.9 else None
        
        # Generate dates
        invoice_date = datetime.now() - timedelta(days=random.randint(1, 60))
        due_date = invoice_date + timedelta(days=30)
        
        # Generate line items
        num_items = random.randint(1, 5)
        line_items = []
        item_total = 0
        
        for j in range(num_items):
            qty = random.randint(1, 20)
            rate = random.uniform(100, 5000)
            item_amount = qty * rate
            item_total += item_amount
            
            line_items.append({
                'description': f'Product {chr(65 + j)}',
                'quantity': qty,
                'rate': round(rate, 2),
                'amount': round(item_amount, 2)
            })
        
        # Calculate totals
        net_amount = round(item_total, 2)
        tax_amount = round(net_amount * 0.18, 2)
        total_amount = round(net_amount + tax_amount, 2)
        
        invoice_data = {
            'invoice_id': f'INV-{2024}{str(i+1).zfill(3)}',
            'vendor_name': vendor,
            'total_amount': total_amount,
            'currency': 'INR',
            'po_number': po_number,
            'due_date': due_date.strftime('%Y-%m-%d'),
            'invoice_date': invoice_date.strftime('%Y-%m-%d'),
            'line_items': line_items,
            'tax_amount': tax_amount,
            'net_amount': net_amount
        }
        
        # Generate PDF
        filename = f'./incoming_invoices/{invoice_data["invoice_id"]}.pdf'
        generate_sample_invoice_pdf(filename, invoice_data)
        
        sample_invoices.append(invoice_data)
    
    # Save sample data as JSON for reference
    with open('./data/sample_invoices.json', 'w') as f:
        json.dump(sample_invoices, f, indent=2, default=str)
    
    print(f"Created {len(sample_invoices)} sample invoices")
    return sample_invoices

def initialize_system():
    """Initialize the system with sample data and configurations"""
    
    print("Initializing Invoice Compliance Tracker...")
    
    # Create sample invoices
    create_sample_invoices()
    
    # Initialize database (this will be done by backend)
    print("Database initialization will be handled by backend.py")
    
    # Create configuration summary
    config_summary = {
        'max_amount': 50000,
        'approved_vendors': ['ABC Pvt Ltd', 'XYZ Logistics'],
        'currency': 'INR',
        'tax_rate': 0.18,
        'default_payment_terms': 30,
        'initialized_at': datetime.now().isoformat()
    }
    
    os.makedirs('./data', exist_ok=True)
    with open('./data/config.json', 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print("System initialization complete!")
    print("\nTo start the application:")
    print("1. pip install -r requirements.txt")
    print("2. Update .env file with your API keys")
    print("3. streamlit run app.py")

def run_system_tests():
    """Run basic system tests"""
    
    print("Running system tests...")
    
    # Test 1: Check if all required directories exist
    required_dirs = [
        './incoming_invoices',
        './processed_invoices',
        './data',
        './logs',
        './exports'
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Missing directory: {directory}")
    
    # Test 2: Check if sample invoices were created
    invoice_files = [f for f in os.listdir('./incoming_invoices') if f.endswith('.pdf')]
    print(f"✅ Found {len(invoice_files)} sample invoice files")
    
    # Test 3: Check environment file
    if os.path.exists('.env'):
        print("✅ Environment file exists")
    else:
        print("❌ Environment file missing")
    
    # Test 4: Check requirements file
    if os.path.exists('requirements.txt'):
        print("✅ Requirements file exists")
    else:
        print("❌ Requirements file missing")
    
    print("System tests complete!")

def export_sample_data():
    """Export sample invoice data for analysis"""
    
    if os.path.exists('./data/sample_invoices.json'):
        with open('./data/sample_invoices.json', 'r') as f:
            sample_data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(sample_data)
        
        # Add some analysis
        print("\nSample Data Analysis:")
        print(f"Total Invoices: {len(df)}")
        print(f"Total Amount: ₹{df['total_amount'].sum():,.2f}")
        print(f"Average Amount: ₹{df['total_amount'].mean():,.2f}")
        print("\nVendor Distribution:")
        print(df['vendor_name'].value_counts())
        
        # Export to CSV
        df.to_csv('./data/sample_invoices.csv', index=False)
        print("\nExported sample data to ./data/sample_invoices.csv")
        
        return df
    else:
        print("No sample data found. Run create_sample_invoices() first.")
        return None

if __name__ == "__main__":
    # Run initialization
    initialize_system()
    
    # Run tests
    run_system_tests()
    
    # Export sample data
    export_sample_data()
