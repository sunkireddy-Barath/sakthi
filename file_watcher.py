"""
File Watcher Module for Automated Invoice Compliance Tracker

This module monitors the incoming_invoices folder for new PDF files and 
automatically processes them through the invoice compliance pipeline.

Author: AI Assistant
Date: September 2025
"""

import os
import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from loguru import logger

# Import our backend processor
from backend import invoice_processor

class InvoiceFileHandler(FileSystemEventHandler):
    """
    File system event handler for processing new invoice PDFs
    """
    
    def __init__(self, processor):
        """
        Initialize the file handler
        
        Args:
            processor: Instance of InvoiceProcessor
        """
        self.processor = processor
        self.processing_queue = []
        self.processing_lock = threading.Lock()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        logger.info("Invoice file handler initialized")
    
    def on_created(self, event):
        """
        Handle file creation events
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # Check if it's a PDF file
        if not file_path.lower().endswith('.pdf'):
            return
        
        # Wait a moment to ensure file is fully written
        time.sleep(2)
        
        # Add to processing queue
        with self.processing_lock:
            if file_path not in self.processing_queue:
                self.processing_queue.append(file_path)
                logger.info(f"Added {file_path} to processing queue")
    
    def on_moved(self, event):
        """
        Handle file move events (e.g., when files are renamed or moved into the folder)
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
        
        dest_path = event.dest_path
        
        # Check if it's a PDF file moved into our watch directory
        if not dest_path.lower().endswith('.pdf'):
            return
        
        # Wait a moment to ensure file is fully moved
        time.sleep(2)
        
        # Add to processing queue
        with self.processing_lock:
            if dest_path not in self.processing_queue:
                self.processing_queue.append(dest_path)
                logger.info(f"Added moved file {dest_path} to processing queue")
    
    def _process_queue(self):
        """
        Background thread function to process files in the queue
        """
        while True:
            try:
                # Check if there are files to process
                with self.processing_lock:
                    if not self.processing_queue:
                        time.sleep(1)
                        continue
                    
                    file_path = self.processing_queue.pop(0)
                
                # Verify file still exists and is readable
                if not os.path.exists(file_path):
                    logger.warning(f"File {file_path} no longer exists, skipping")
                    continue
                
                if not os.access(file_path, os.R_OK):
                    logger.warning(f"File {file_path} is not readable, skipping")
                    continue
                
                # Process the invoice
                logger.info(f"Processing invoice: {file_path}")
                
                try:
                    invoice_data, compliance_result = self.processor.process_invoice(file_path)
                    
                    if invoice_data:
                        logger.info(f"Successfully processed {invoice_data.invoice_id} - Status: {compliance_result.status}")
                    else:
                        logger.error(f"Failed to process {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                
            except Exception as e:
                logger.error(f"Error in processing queue: {e}")
                time.sleep(5)  # Wait before retrying

class InvoiceWatcher:
    """
    Main watcher class that monitors the incoming invoices directory
    """
    
    def __init__(self, watch_directory: str = "./incoming_invoices"):
        """
        Initialize the invoice watcher
        
        Args:
            watch_directory: Directory to monitor for new invoice files
        """
        self.watch_directory = os.path.abspath(watch_directory)
        self.observer = None
        self.event_handler = None
        
        # Ensure watch directory exists
        os.makedirs(self.watch_directory, exist_ok=True)
        
        logger.info(f"Invoice watcher initialized for directory: {self.watch_directory}")
    
    def start_watching(self):
        """
        Start monitoring the directory for new files
        """
        try:
            # Create event handler
            self.event_handler = InvoiceFileHandler(invoice_processor)
            
            # Create observer
            self.observer = Observer()
            self.observer.schedule(
                self.event_handler, 
                self.watch_directory, 
                recursive=False
            )
            
            # Start observer
            self.observer.start()
            
            logger.info(f"Started watching directory: {self.watch_directory}")
            
            # Process any existing files in the directory
            self._process_existing_files()
            
        except Exception as e:
            logger.error(f"Failed to start watching: {e}")
            raise
    
    def stop_watching(self):
        """
        Stop monitoring the directory
        """
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
                logger.info("Stopped watching directory")
                
        except Exception as e:
            logger.error(f"Error stopping watcher: {e}")
    
    def _process_existing_files(self):
        """
        Process any PDF files that already exist in the watch directory
        """
        try:
            existing_files = [
                os.path.join(self.watch_directory, f) 
                for f in os.listdir(self.watch_directory) 
                if f.lower().endswith('.pdf')
            ]
            
            if existing_files:
                logger.info(f"Found {len(existing_files)} existing PDF files to process")
                
                # Add existing files to processing queue
                for file_path in existing_files:
                    with self.event_handler.processing_lock:
                        if file_path not in self.event_handler.processing_queue:
                            self.event_handler.processing_queue.append(file_path)
                            logger.info(f"Added existing file {file_path} to processing queue")
            
        except Exception as e:
            logger.error(f"Error processing existing files: {e}")
    
    def get_queue_status(self):
        """
        Get current status of the processing queue
        
        Returns:
            Dictionary with queue information
        """
        try:
            if not self.event_handler:
                return {"queue_length": 0, "files": []}
            
            with self.event_handler.processing_lock:
                return {
                    "queue_length": len(self.event_handler.processing_queue),
                    "files": list(self.event_handler.processing_queue)
                }
                
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {"queue_length": 0, "files": []}

def run_watcher_service():
    """
    Run the file watcher as a service
    This function can be called to start the watcher in the background
    """
    try:
        watcher = InvoiceWatcher()
        watcher.start_watching()
        
        logger.info("Invoice watcher service started. Press Ctrl+C to stop.")
        
        # Keep the service running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping watcher...")
            watcher.stop_watching()
            
    except Exception as e:
        logger.error(f"Watcher service error: {e}")

# Create global watcher instance
invoice_watcher = InvoiceWatcher()

if __name__ == "__main__":
    # Run as standalone service
    run_watcher_service()
