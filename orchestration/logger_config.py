import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):

    os.makedirs(log_dir, exist_ok=True)
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    all_logs_file = os.path.join(log_dir, "pipeline.log")
    file_handler = logging.handlers.RotatingFileHandler(
        all_logs_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    error_logs_file = os.path.join(log_dir, "errors.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_logs_file, maxBytes=10*1024*1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    component_loggers = [
        "data_collection",
        "data_processing", 
        "fine_tuning",
        "evaluation",
        "deployment",
        "orchestration"
    ]
    
    for component in component_loggers:
        component_file = os.path.join(log_dir, f"{component}.log")
        component_handler = logging.handlers.RotatingFileHandler(
            component_file, maxBytes=5*1024*1024, backupCount=3
        )
        component_handler.setLevel(numeric_level)
        component_handler.setFormatter(detailed_formatter)
        
        component_logger = logging.getLogger(component)
        component_logger.addHandler(component_handler)
        component_logger.setLevel(numeric_level)
    
    logging.info("Logging configuration completed")

class PipelineLogger:

    def __init__(self, component_name: str):
        self.logger = logging.getLogger(component_name)
        self.component = component_name
    
    def log_pipeline_start(self, pipeline_type: str):
        self.logger.info(f"Starting {pipeline_type} pipeline")
    
    def log_pipeline_end(self, pipeline_type: str, success: bool, duration: float = None):
        status = "SUCCESS" if success else "FAILED"
        duration_str = f" (Duration: {duration:.2f}s)" if duration else ""
        self.logger.info(f"{pipeline_type} pipeline {status}{duration_str}")
    
    def log_step(self, step_name: str, status: str, details: str = None):
        message = f"Step '{step_name}': {status}"
        if details:
            message += f" - {details}"
        self.logger.info(message)
    
    def log_metrics(self, metrics: dict):
        self.logger.info(f"Metrics: {metrics}")
    
    def log_error(self, error_msg: str, exception: Exception = None):
        if exception:
            self.logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(error_msg)

