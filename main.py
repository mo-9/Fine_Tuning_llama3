"""
AI Engineer Pipeline - Main Entry Point

This script provides a command-line interface for running the complete
end-to-end pipeline for fine-tuning and deploying language models.

Usage:
    python main.py --help
    python main.py run-pipeline
    python main.py start-api
    python main.py collect-data
    python main.py train-model
    python main.py evaluate-model
    python main.py deploy-model
"""

import argparse
import logging
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from orchestration.pipeline_orchestrator import PipelineOrchestrator
from orchestration.scheduler import PipelineScheduler
from orchestration.logger_config import setup_logging
from deployment.api_server import app
import uvicorn

def setup_environment():
    directories = ["logs", "data", "results", "model_registry"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    setup_logging(log_level="INFO")
    
    logging.info("Environment setup completed")

def run_full_pipeline():
    logging.info("Starting full pipeline execution")
    
    orchestrator = PipelineOrchestrator()
    results = orchestrator.run_full_pipeline()
    
    if results["success"]:
        logging.info("Pipeline execution completed successfully")
        print(" Pipeline execution completed successfully!")
        print(f" Results: {results}")
    else:
        logging.error("Pipeline execution failed")
        print(" Pipeline execution failed!")
        print(f" Results: {results}")
        sys.exit(1)

def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    logging.info(f"Starting API server on {host}:{port}")
    print(f" Starting API server on http://{host}:{port}")
    print(" API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(app, host=host, port=port, log_level="info")

def collect_data():
    logging.info("Starting data collection")
    
    orchestrator = PipelineOrchestrator()
    success = orchestrator.run_data_collection()
    
    if success:
        print(" Data collection completed successfully!")
    else:
        print(" Data collection failed!")
        sys.exit(1)

def train_model():
    logging.info("Starting model training")
    
    orchestrator = PipelineOrchestrator()
    success = orchestrator.run_training()
    
    if success:
        print(" Model training completed successfully!")
    else:
        print("Model training failed!")
        sys.exit(1)

def evaluate_model():
    logging.info("Starting model evaluation")
    
    orchestrator = PipelineOrchestrator()
    results = orchestrator.run_evaluation()
    
    if results:
        print(" Model evaluation completed successfully!")
        print(f" Evaluation results: {results}")
    else:
        print(" Model evaluation failed!")
        sys.exit(1)

def deploy_model():
    logging.info("Starting model deployment")
    
    orchestrator = PipelineOrchestrator()
    success = orchestrator.run_deployment()
    
    if success:
        print("Model deployment completed successfully!")
    else:
        print(" Model deployment failed!")
        sys.exit(1)

def start_scheduler():
    logging.info("Starting pipeline scheduler")
    
    scheduler = PipelineScheduler()
    
    scheduler.schedule_daily_training(hour=2, minute=0)
    scheduler.schedule_weekly_full_pipeline(day="sunday", hour=1, minute=0)
    scheduler.schedule_hourly_monitoring()
    
    print(" Pipeline scheduler started with default schedules:")
    print("  - Daily training at 02:00")
    print("  - Weekly full pipeline on Sunday at 01:00")
    print("  - Hourly monitoring checks")
    print("Press Ctrl+C to stop the scheduler")
    
    try:
        scheduler.start_scheduler()
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n Stopping scheduler...")
        scheduler.stop_scheduler()
        print("✅ Scheduler stopped")

def main():
    parser = argparse.ArgumentParser(
        description="AI Engineer Pipeline - End-to-End LLM Fine-Tuning and Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run-pipeline          # Run the complete pipeline
  python main.py start-api             # Start the API server
  python main.py collect-data          # Run data collection only
  python main.py train-model           # Run model training only
  python main.py evaluate-model        # Run model evaluation only
  python main.py deploy-model          # Run model deployment only
  python main.py start-scheduler       # Start the pipeline scheduler
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("run-pipeline", help="Run the complete end-to-end pipeline")
    
    api_parser = subparsers.add_parser("start-api", help="Start the FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    
    # Individual pipeline components
    subparsers.add_parser("collect-data", help="Run data collection pipeline")
    subparsers.add_parser("train-model", help="Run model training pipeline")
    subparsers.add_parser("evaluate-model", help="Run model evaluation pipeline")
    subparsers.add_parser("deploy-model", help="Run model deployment pipeline")
    
    subparsers.add_parser("start-scheduler", help="Start the pipeline scheduler")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_environment()
    
    try:
        if args.command == "run-pipeline":
            run_full_pipeline()
        elif args.command == "start-api":
            start_api_server(args.host, args.port)
        elif args.command == "collect-data":
            collect_data()
        elif args.command == "train-model":
            train_model()
        elif args.command == "evaluate-model":
            evaluate_model()
        elif args.command == "deploy-model":
            deploy_model()
        elif args.command == "start-scheduler":
            start_scheduler()
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f" Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

