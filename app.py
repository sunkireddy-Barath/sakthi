"""
Streamlit Dashboard for Automated Invoice Compliance Tracker

This is the main web interface that provides:
- Real-time invoice tracking and monitoring
- Compliance metrics and analytics
- Interactive chat interface for querying invoice data
- Export and reporting functionality
- File upload and manual processing

Author: Development Team
Date: September 2025
"""

# Configure matplotlib backend BEFORE any other imports
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

# Set environment variable to prevent OpenGL issues
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time
from datetime import datetime, timedelta
from io import BytesIO
import base64

# Import our backend modules with detailed error handling
try:
    # First test individual imports
    import backend
    
    from backend import invoice_processor
    
    from file_watcher import invoice_watcher
    
except ImportError as e:
    st.error(f"Failed to import backend modules: {e}")
    st.error("This might be due to missing system dependencies. Check the deployment logs.")
    
    # Show what we can about the error
    import traceback
    st.code(traceback.format_exc())
    
    # Create a dummy processor for testing
    st.warning("Running in fallback mode - some features may not work")
    invoice_processor = None
    invoice_watcher = None

# Page configuration
st.set_page_config(
    page_title="Invoice Compliance Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-approved {
        color: #28a745;
        font-weight: bold;
    }
    .status-flagged {
        color: #dc3545;
        font-weight: bold;
    }
    .status-pending {
        color: #ffc107;
        font-weight: bold;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'invoices_df' not in st.session_state:
        st.session_state.invoices_df = pd.DataFrame()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    if 'selected_status_filter' not in st.session_state:
        st.session_state.selected_status_filter = "All"
    
    if 'dashboard_metrics' not in st.session_state:
        st.session_state.dashboard_metrics = {}
    
    if 'last_metrics_refresh' not in st.session_state:
        st.session_state.last_metrics_refresh = None

@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_invoice_data_cached():
    """Load invoice data with caching for better performance"""
    try:
        df = invoice_processor.get_all_invoices()
        return df
    except Exception as e:
        st.error(f"Failed to load invoice data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)  # Cache metrics for 1 minute
def get_dashboard_metrics_cached():
    """Get dashboard metrics with caching"""
    try:
        return invoice_processor.get_dashboard_metrics()
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        return {}

def load_invoice_data():
    """Load invoice data from backend and update session state"""
    df = load_invoice_data_cached()
    st.session_state.invoices_df = df
    st.session_state.last_refresh = datetime.now()
    return df

def format_currency(amount, currency="INR"):
    """Format currency amounts for display"""
    if pd.isna(amount):
        return "N/A"
    return f"{currency} {amount:,.2f}"

def get_status_color(status):
    """Get color for status display"""
    colors = {
        'approved': '#28a745',
        'flagged': '#dc3545',
        'pending': '#ffc107',
        'error': '#6c757d'
    }
    return colors.get(status, '#6c757d')

def render_header():
    """Render the main header and navigation"""
    st.markdown('<h1 class="main-header">Automated Invoice Compliance Tracker</h1>', 
                unsafe_allow_html=True)
    
    # Add subtitle and description
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3 style="color: #666; font-weight: normal;">Smart Invoice Processing & Compliance Management System</h3>
        <p style="color: #888; font-size: 1.1rem;">
            Streamline your invoice workflow with intelligent automation, real-time compliance checking, 
            and advanced analytics powered by modern technology.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add navigation tabs
    tabs = st.tabs(["Dashboard", "Invoice Details", "Smart Chat", "Settings", "Export"])
    return tabs

def render_sidebar():
    """Render the sidebar with controls and metrics"""
    st.sidebar.header("Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh data", 
        value=st.session_state.auto_refresh,
        help="Automatically refresh data every 30 seconds"
    )
    st.session_state.auto_refresh = auto_refresh
    
    # Manual refresh button
    if st.sidebar.button("Refresh Now"):
        load_invoice_data()
        st.rerun()
    
    # System status section
    st.sidebar.subheader("System Status")
    
    # Database status
    try:
        db_status = "Connected" if invoice_processor else "Error"
    except:
        db_status = "Error"
    st.sidebar.write(f"**Database:** {db_status}")
    
    # File watcher status
    st.sidebar.subheader("File Processing Status")
    try:
        queue_status = invoice_watcher.get_queue_status()
        st.sidebar.metric("Files in Queue", queue_status['queue_length'])
        if queue_status['files']:
            st.sidebar.write("Processing:")
            for file_path in queue_status['files'][:3]:  # Show max 3 files
                filename = os.path.basename(file_path)
                st.sidebar.write(f"• {filename}")
    except Exception as e:
        st.sidebar.error(f"Watcher status unavailable: {e}")
    
    # Status filter
    st.sidebar.subheader("Filters")
    status_options = ["All", "approved", "flagged", "pending", "error"]
    selected_status = st.sidebar.selectbox(
        "Filter by Status",
        status_options,
        index=status_options.index(st.session_state.selected_status_filter)
    )
    st.session_state.selected_status_filter = selected_status
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        help="Filter invoices by processing date"
    )
    
    return selected_status, date_range

def render_metrics_dashboard(df):
    """Render the main metrics dashboard"""
    if df.empty:
        # Welcome section with project information
        st.markdown("""
        ## Welcome to the Automated Invoice Compliance Tracker!
        
        ### Smart Invoice Processing System
        
        This advanced application leverages modern technology to streamline your invoice management workflow.
        """)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Smart Features
            - **Intelligent Data Extraction**
            - Smart PDF Processing
            - Natural Language Chat
            - Automated Compliance
            """)
        
        with col2:
            st.markdown("""
            ### Analytics
            - Real-time Dashboards
            - Compliance Scoring
            - Fraud Detection
            - Anomaly Analysis
            """)
        
        with col3:
            st.markdown("""
            ### Automation
            - Auto File Monitoring
            - Instant Processing
            - Email Notifications
            - Smart Search
            """)
        
        st.markdown("---")
        
        # Quick start guide
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload Invoices**: Drop PDF files into the `incoming_invoices/` folder
        2. **Auto Processing**: Watch the system extract data in real-time
        3. **Chat Interface**: Ask questions about your invoices
        4. **View Analytics**: Monitor compliance and performance
        5. **Export Reports**: Download insights and data
        """)
        
        # Sample data generation
        st.markdown("### Try it Now!")
        
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            if st.button("Generate Sample Data", type="primary", key="generate_sample_data"):
                with st.spinner("Generating sample invoices..."):
                    try:
                        from utils import create_sample_invoices
                        create_sample_invoices(5)
                        st.success("Generated 5 sample invoices! Refresh to see them.")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating samples: {e}")
        
        with col_demo2:
            st.markdown("""
            **Sample Features:**
            - Realistic invoice data
            - Various compliance scenarios
            - Multiple vendor types
            - Different amount ranges
            """)
        
        # Technology stack
        st.markdown("---")
        st.markdown("### Technology Stack")
        
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            st.markdown("""
            **Backend**
            - Python 3.9+
            - Smart Data Processing
            - Machine Learning
            - SQLite
            """)
        
        with tech_col2:
            st.markdown("""
            **Frontend**
            - Streamlit
            - Interactive Visualizations
            - Responsive UI
            - Real-time Updates
            """)
        
        with tech_col3:
            st.markdown("""
            **Database**
            - Vector Store
            - SQLite RDBMS
            - Smart Search System
            - Semantic Search
            """)
        
        with tech_col4:
            st.markdown("""
            **Deployment**
            - Docker Ready
            - Cloud Config
            - Auto Scaling
            - CI/CD Pipeline
            """)
        
        return
    
    # Get dashboard metrics with caching
    metrics = get_dashboard_metrics_cached()
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Invoices",
            value=metrics.get('total_invoices', 0)
        )
    
    with col2:
        st.metric(
            label="Approved",
            value=metrics.get('approved_count', 0),
            delta=f"{(metrics.get('approved_count', 0) / max(metrics.get('total_invoices', 1), 1) * 100):.1f}%"
        )
    
    with col3:
        st.metric(
            label="Flagged",
            value=metrics.get('flagged_count', 0),
            delta=f"{(metrics.get('flagged_count', 0) / max(metrics.get('total_invoices', 1), 1) * 100):.1f}%"
        )
    
    with col4:
        st.metric(
            label="Total Amount",
            value=format_currency(metrics.get('total_amount', 0))
        )
    
    # Analytics section
    st.subheader("Analytics")
    
    analytics_col1, analytics_col2 = st.columns(2)
    
    with analytics_col1:
        # Invoice status overview
        if not df.empty:
            status_counts = df['status'].value_counts()
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Invoice Status Overview",
                color_discrete_map={
                    'approved': '#28a745',
                    'flagged': '#dc3545',
                    'pending': '#ffc107',
                    'error': '#6c757d'
                }
            )
            st.plotly_chart(fig_pie, width='stretch')
    
    with analytics_col2:
        # Compliance score analysis
        if not df.empty and 'compliance_score' in df.columns:
            fig_hist = px.histogram(
                df,
                x='compliance_score',
                title="Compliance Score Analysis",
                nbins=20,
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.update_layout(
                xaxis_title="Compliance Score",
                yaxis_title="Number of Invoices"
            )
            st.plotly_chart(fig_hist, width='stretch')
    
    # Processing timeline
    if not df.empty and 'processed_at' in df.columns:
        st.subheader("Processing Timeline")
        
        # Convert processed_at to datetime
        df_time = df.copy()
        df_time['processed_date'] = pd.to_datetime(df_time['processed_at']).dt.date
        
        # Group by date and status
        time_data = df_time.groupby(['processed_date', 'status']).size().reset_index(name='count')
        
        fig_timeline = px.line(
            time_data,
            x='processed_date',
            y='count',
            color='status',
            title="Daily Invoice Processing Activity",
            color_discrete_map={
                'approved': '#28a745',
                'flagged': '#dc3545',
                'pending': '#ffc107',
                'error': '#6c757d'
            }
        )
        st.plotly_chart(fig_timeline, width='stretch')
    
    # Top vendors analysis
    if not df.empty:
        st.subheader("Top Vendors")
        vendor_data = df.groupby('vendor_name').agg({
            'total_amount': 'sum',
            'invoice_id': 'count'
        }).reset_index()
        vendor_data.columns = ['Vendor', 'Total Amount', 'Invoice Count']
        vendor_data = vendor_data.sort_values('Total Amount', ascending=False).head(10)
        
        fig_vendors = px.bar(
            vendor_data,
            x='Vendor',
            y='Total Amount',
            title="Top 10 Vendors by Total Amount",
            color='Invoice Count',
            color_continuous_scale='Blues'
        )
        fig_vendors.update_xaxes(tickangle=45)
        st.plotly_chart(fig_vendors, width='stretch')

def render_invoice_table(df, status_filter):
    """Render the detailed invoice table"""
    st.subheader("Invoice Details")
    
    if df.empty:
        st.info("No invoices to display.")
        return
    
    # Apply status filter
    if status_filter != "All":
        df_filtered = df[df['status'] == status_filter]
    else:
        df_filtered = df
    
    if df_filtered.empty:
        st.info(f"No invoices with status '{status_filter}'.")
        return
    
    # Display summary
    st.write(f"Showing {len(df_filtered)} of {len(df)} invoices")
    
    # Prepare display dataframe
    display_df = df_filtered[[
        'invoice_id', 'vendor_name', 'total_amount', 'currency', 
        'po_number', 'due_date', 'status', 'compliance_score',
        'processed_at'
    ]].copy()
    
    # Format columns for better display
    display_df['total_amount'] = display_df.apply(
        lambda row: format_currency(row['total_amount'], row['currency']), 
        axis=1
    )
    display_df['compliance_score'] = display_df['compliance_score'].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    display_df['processed_at'] = pd.to_datetime(display_df['processed_at']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Rename columns for display
    display_df.columns = [
        'Invoice ID', 'Vendor', 'Amount', 'Currency', 
        'PO Number', 'Due Date', 'Status', 'Compliance Score', 'Processed'
    ]
    
    # Display table with conditional formatting
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True
    )
    
    # Show detailed view for selected invoice
    st.subheader("Invoice Details")
    
    if not df_filtered.empty:
        invoice_ids = df_filtered['invoice_id'].tolist()
        selected_invoice = st.selectbox("Select Invoice for Details", ["None"] + invoice_ids, key="select_invoice_details")
        
        if selected_invoice != "None":
            invoice_details = df_filtered[df_filtered['invoice_id'] == selected_invoice].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"**Invoice ID:** {invoice_details['invoice_id']}")
                st.write(f"**Vendor:** {invoice_details['vendor_name']}")
                st.write(f"**Amount:** {format_currency(invoice_details['total_amount'], invoice_details['currency'])}")
                st.write(f"**PO Number:** {invoice_details['po_number'] or 'N/A'}")
                st.write(f"**Due Date:** {invoice_details['due_date'] or 'N/A'}")
                
            with col2:
                st.write("**Compliance Information:**")
                st.write(f"**Status:** {invoice_details['status']}")
                st.write(f"**Compliance Score:** {invoice_details['compliance_score']:.2f}")
                
                if 'anomaly_score' in invoice_details and pd.notna(invoice_details['anomaly_score']):
                    st.write(f"**Anomaly Score:** {invoice_details['anomaly_score']:.3f}")
                
                if 'fraud_score' in invoice_details and pd.notna(invoice_details['fraud_score']):
                    st.write(f"**Fraud Score:** {invoice_details['fraud_score']:.3f}")
            
            # Show reasons if flagged
            if invoice_details['status'] == 'flagged' and 'reasons' in invoice_details:
                st.write("**Flagged Reasons:**")
                try:
                    reasons = json.loads(invoice_details['reasons']) if isinstance(invoice_details['reasons'], str) else invoice_details['reasons']
                    for reason in reasons:
                        st.write(f"• {reason}")
                except:
                    st.write("• Compliance check failed")
            
            # Show line items if available
            if 'line_items' in invoice_details and invoice_details['line_items']:
                st.write("**Line Items:**")
                try:
                    line_items = json.loads(invoice_details['line_items']) if isinstance(invoice_details['line_items'], str) else invoice_details['line_items']
                    if line_items:
                        line_items_df = pd.DataFrame(line_items)
                        st.dataframe(line_items_df, width='stretch')
                except:
                    st.write("Line items could not be displayed")

def render_chat_interface():
    """Render the smart chat interface for querying invoice data"""
    st.subheader("Smart Assistant - Ask about your invoices")
    
    # Chat input
    user_question = st.text_input(
        "Ask a question about your invoices:",
        placeholder="e.g., Why was invoice INV-001 flagged? or Show me all invoices from ABC Pvt Ltd",
        key="chat_input_question"
    )
    
    if st.button("Send", key="send_chat_message") and user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'type': 'user',
            'message': user_question,
            'timestamp': datetime.now()
        })
        
        # Get smart response
        with st.spinner("Thinking..."):
            try:
                smart_response = invoice_processor.answer_question_with_rag(user_question)
                
                # Add smart response to chat history
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'message': smart_response,
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'message': error_message,
                    'timestamp': datetime.now()
                })
    
    # Display chat history
    if st.session_state.chat_history:
        st.write("**Conversation:**")
        
        # Create a container for the chat
        chat_container = st.container()
        
        with chat_container:
            for chat in reversed(st.session_state.chat_history[-10:]):  # Show last 10 messages
                timestamp = chat['timestamp'].strftime("%H:%M:%S")
                
                if chat['type'] == 'user':
                    st.write(f"**You** ({timestamp}): {chat['message']}")
                else:
                    st.write(f"**Smart Assistant** ({timestamp}): {chat['message']}")
                st.write("---")
    
    # Clear chat button
    if st.button("Clear Chat History", key="clear_chat_history_main"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Suggested questions
    st.write("**Suggested questions:**")
    suggestions = [
        "How many invoices are flagged?",
        "What is the total amount of approved invoices?",
        "Show me invoices from the last week",
        "Which vendor has the most invoices?",
        "Why are invoices getting flagged?"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggestion_{suggestion}"):
            # Simulate clicking with the suggestion
            st.session_state.temp_question = suggestion
            st.rerun()

def render_settings():
    """Render the settings and configuration page"""
    st.subheader("Settings & Configuration")
    
    # File upload section
    st.write("**Manual Invoice Upload**")
    uploaded_files = st.file_uploader(
        "Upload PDF invoices",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF files to process manually"
    )
    
    if uploaded_files:
        if st.button("Process Uploaded Files", key="process_uploaded_files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                temp_path = f"./incoming_invoices/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                try:
                    # Process the invoice
                    invoice_data, compliance_result = invoice_processor.process_invoice(temp_path)
                    
                    if invoice_data:
                        st.success(f"Processed {uploaded_file.name} - Status: {compliance_result.status}")
                    else:
                        st.error(f"Failed to process {uploaded_file.name}")
                        
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            
            # Refresh data
            load_invoice_data()
    
    # Configuration section
    st.write("**Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Compliance Rules:**")
        st.info(f"Max Amount: ₹{invoice_processor.max_amount:,.2f}")
        st.info(f"Approved Vendors: {', '.join(invoice_processor.approved_vendors)}")
    
    with col2:
        st.write("**System Status:**")
        try:
            # Check database connection
            test_df = invoice_processor.get_all_invoices()
            st.success("Database Connected")
        except:
            st.error("Database Connection Failed")
        
        # Check ChromaDB
        try:
            if invoice_processor.collection:
                st.success("Vector Database Connected")
            else:
                st.warning("Vector Database Not Available")
        except:
            st.error("Vector Database Connection Failed")
    
    # Data management
    st.write("**Data Management**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Reload All Data", key="reload_data_settings"):
            load_invoice_data()
            st.success("Data reloaded successfully!")
    
    with col2:
        if st.button("Clear Chat History", key="clear_chat_history_settings"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
    
    with col3:
        if st.button("Reset Database", type="secondary", key="reset_database_settings"):
            if st.checkbox("I understand this will delete all data", key="confirm_reset_database"):
                # This would be dangerous in production
                st.warning("This feature is disabled for safety")

def render_export_page(df=None):
    """Render the export and reporting page"""
    st.subheader("Export & Reports")
    
    # Use provided dataframe or load from session state
    if df is None:
        df = st.session_state.invoices_df
    
    # Export options
    st.write("**Data Export**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "Excel", "JSON"]
        )
        
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "approved", "flagged", "pending", "error"]
        )
    
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now())
        )
        
        include_details = st.checkbox(
            "Include detailed information",
            value=True,
            key="export_include_details",
            help="Include line items, extracted text, and other detailed data"
        )
    
    if st.button("Export Data", key="export_data_button"):
        try:
            with st.spinner("Preparing export..."):
                # Get filtered data
                df = st.session_state.invoices_df
                
                if not df.empty:
                    # Apply filters
                    if status_filter != "All":
                        df = df[df['status'] == status_filter]
                    
                    # Apply date filter
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        df['processed_date'] = pd.to_datetime(df['processed_at']).dt.date
                        df = df[(df['processed_date'] >= start_date) & (df['processed_date'] <= end_date)]
                    
                    if df.empty:
                        st.warning("No data matches the selected filters.")
                    else:
                        # Prepare export
                        if export_format == "CSV":
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"invoices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        elif export_format == "Excel":
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False, sheet_name='Invoices')
                            
                            st.download_button(
                                label="Download Excel",
                                data=buffer.getvalue(),
                                file_name=f"invoices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        elif export_format == "JSON":
                            json_data = df.to_json(orient='records', indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"invoices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        st.success(f"Exported {len(df)} invoices")
                else:
                    st.warning("No data available for export")
                    
        except Exception as e:
            st.error(f"Export failed: {e}")
    
    # Report generation
    st.write("**Reports**")
    
    report_type = st.selectbox(
        "Report Type",
        ["Compliance Summary", "Vendor Analysis", "Fraud Detection Summary", "Processing Timeline"]
    )
    
    if st.button("Generate Report", key="generate_report_button"):
        try:
            df = st.session_state.invoices_df
            
            if df.empty:
                st.warning("No data available for reporting")
            else:
                if report_type == "Compliance Summary":
                    st.write("### Compliance Summary Report")
                    
                    # Overall metrics
                    total_invoices = len(df)
                    approved_count = len(df[df['status'] == 'approved'])
                    flagged_count = len(df[df['status'] == 'flagged'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Invoices", total_invoices)
                    with col2:
                        st.metric("Approval Rate", f"{(approved_count/total_invoices*100):.1f}%")
                    with col3:
                        st.metric("Flag Rate", f"{(flagged_count/total_invoices*100):.1f}%")
                    
                    # Detailed breakdown
                    if 'compliance_score' in df.columns:
                        avg_score = df['compliance_score'].mean()
                        st.write(f"**Average Compliance Score:** {avg_score:.2f}")
                    
                    # Common flag reasons
                    if flagged_count > 0:
                        st.write("**Most Common Flag Reasons:**")
                        flagged_df = df[df['status'] == 'flagged']
                        # This would need more sophisticated parsing of reasons
                        st.write("(Detailed reason analysis would be implemented here)")
                
                elif report_type == "Vendor Analysis":
                    st.write("### Vendor Analysis Report")
                    
                    vendor_stats = df.groupby('vendor_name').agg({
                        'total_amount': ['sum', 'mean', 'count'],
                        'compliance_score': 'mean'
                    }).round(2)
                    
                    vendor_stats.columns = ['Total Amount', 'Average Amount', 'Invoice Count', 'Avg Compliance Score']
                    vendor_stats = vendor_stats.sort_values('Total Amount', ascending=False)
                    
                    st.dataframe(vendor_stats, width='stretch')
                
                # Add more report types as needed
                
        except Exception as e:
            st.error(f"Report generation failed: {e}")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Start file watcher if not already running
    try:
        if not invoice_watcher.observer or not invoice_watcher.observer.is_alive():
            invoice_watcher.start_watching()
    except Exception as e:
        st.error(f"Failed to start file watcher: {e}")
    
    # Render sidebar
    status_filter, date_range = render_sidebar()
    
    # Auto-refresh data only if on a tab that needs it
    should_load_data = False
    
    # Check if we need to load data based on which tab might be active
    if st.session_state.auto_refresh:
        current_time = datetime.now()
        if (current_time - st.session_state.last_refresh).seconds > 30:  # Refresh every 30 seconds
            should_load_data = True
    
    # Load initial data only if needed and not already available
    if st.session_state.invoices_df.empty and should_load_data:
        with st.spinner("Loading invoice data..."):
            load_invoice_data()
    
    # Render main content
    tabs = render_header()
    
    with tabs[0]:  # Dashboard
        # Load data only when dashboard tab is active
        if st.session_state.invoices_df.empty:
            with st.spinner("Loading dashboard data..."):
                load_invoice_data()
        render_metrics_dashboard(st.session_state.invoices_df)
    
    with tabs[1]:  # Invoice Details
        # Load data only when details tab is active  
        if st.session_state.invoices_df.empty:
            with st.spinner("Loading invoice details..."):
                load_invoice_data()
        render_invoice_table(st.session_state.invoices_df, status_filter)
    
    with tabs[2]:  # Smart Chat
        render_chat_interface()
    
    with tabs[3]:  # Settings
        render_settings()
    
    with tabs[4]:  # Export
        # Load data only when export tab is active
        if st.session_state.invoices_df.empty:
            with st.spinner("Loading export data..."):
                load_invoice_data()
        render_export_page(st.session_state.invoices_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Automated Invoice Compliance Tracker** | "
        f"Last Updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Total Invoices: {len(st.session_state.invoices_df)}"
    )

if __name__ == "__main__":
    main()
