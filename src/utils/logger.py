#!/usr/bin/env python3
"""
Logging utility for the Agentic Architecture System
Provides consistent logging across all components
"""

import logging
import sys
from datetime import datetime
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name and level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one
    """
    return logging.getLogger(name)

class LoggerMixin:
    """
    Mixin class to add logging capability to other classes
    """
    
    def __init__(self, name: Optional[str] = None):
        self.logger_name = name or self.__class__.__name__
        self.logger = setup_logger(self.logger_name)
    
    def log_info(self, message: str):
        """Log an info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log an error message"""
        self.logger.error(message)
    
    def log_debug(self, message: str):
        """Log a debug message"""
        self.logger.debug(message)
    
    def log_critical(self, message: str):
        """Log a critical message"""
        self.logger.critical(message)