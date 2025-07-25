import schedule
import time
import logging
from typing import Callable, Dict, List
from datetime import datetime
import threading
from .pipeline_orchestrator import PipelineOrchestrator

class PipelineScheduler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.orchestrator = PipelineOrchestrator()
        self.running = False
        self.scheduler_thread = None
        
    def schedule_daily_training(self, hour: int = 2, minute: int = 0):
        schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(self._run_training_job)
        self.logger.info(f"Scheduled daily training at {hour:02d}:{minute:02d}")
    
    def schedule_weekly_full_pipeline(self, day: str = "sunday", hour: int = 1, minute: int = 0):
        getattr(schedule.every(), day.lower()).at(f"{hour:02d}:{minute:02d}").do(self._run_full_pipeline_job)
        self.logger.info(f"Scheduled weekly full pipeline on {day} at {hour:02d}:{minute:02d}")
    
    def schedule_hourly_monitoring(self):
        schedule.every().hour.do(self._run_monitoring_job)
        self.logger.info("Scheduled hourly monitoring")
    
    def _run_training_job(self):
        self.logger.info("Executing scheduled training job")
        try:
            result = self.orchestrator.run_training()
            self.logger.info(f"Scheduled training completed: {result}")
        except Exception as e:
            self.logger.error(f"Scheduled training failed: {str(e)}")
    
    def _run_full_pipeline_job(self):
        self.logger.info("Executing scheduled full pipeline job")
        try:
            results = self.orchestrator.run_full_pipeline()
            self.logger.info(f"Scheduled full pipeline completed: {results['success']}")
        except Exception as e:
            self.logger.error(f"Scheduled full pipeline failed: {str(e)}")
    
    def _run_monitoring_job(self):
        self.logger.info("Executing scheduled monitoring job")
        # Add monitoring logic here
        # For example, check system health, model performance, etc.
    
    def start_scheduler(self):
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        self.logger.info("Scheduler started")
    
    def stop_scheduler(self):
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        self.logger.info("Scheduler stopped")
    
    def _run_scheduler(self):
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def trigger_manual_training(self) -> Dict:
        self.logger.info("Manual training triggered")
        try:
            result = self.orchestrator.run_training()
            return {"success": result, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Manual training failed: {str(e)}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}
    
    def trigger_manual_full_pipeline(self) -> Dict:
        self.logger.info("Manual full pipeline triggered")
        try:
            results = self.orchestrator.run_full_pipeline()
            return results
        except Exception as e:
            self.logger.error(f"Manual full pipeline failed: {str(e)}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}
    
    def get_scheduled_jobs(self) -> List[Dict]:
        jobs = []
        for job in schedule.jobs:
            jobs.append({
                "job": str(job.job_func),
                "next_run": str(job.next_run),
                "interval": str(job.interval),
                "unit": job.unit
            })
        return jobs

