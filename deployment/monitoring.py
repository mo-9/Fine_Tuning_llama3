import time
import logging
from typing import Dict, List
from collections import defaultdict, deque
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading

class APIMonitor:
    def __init__(self, metrics_port: int = 8001):
        self.metrics_port = metrics_port
        self.logger = logging.getLogger(__name__)
        
        # Prometheus metrics
        self.request_count = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.active_connections = Gauge(
            'api_active_connections',
            'Number of active connections'
        )
        
        self.model_inference_time = Histogram(
            'model_inference_duration_seconds',
            'Model inference duration in seconds'
        )
        
        self.model_requests = Counter(
            'model_requests_total',
            'Total number of model inference requests'
        )
        
        # In-memory metrics for simple monitoring
        self.request_times = deque(maxlen=1000)
        self.error_count = defaultdict(int)
        self.endpoint_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
        
        # Start Prometheus metrics server
        self._start_metrics_server()

    def _start_metrics_server(self):
        """Start the Prometheus metrics server."""
        try:
            start_http_server(self.metrics_port)
            self.logger.info(f"Metrics server started on port {self.metrics_port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")

    def record_request(self, method: str, endpoint: str, status_code: int, 
                      duration: float):
        """Record API request metrics."""
        # Prometheus metrics
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # In-memory metrics
        self.request_times.append(duration)
        self.endpoint_stats[endpoint]['count'] += 1
        self.endpoint_stats[endpoint]['total_time'] += duration
        
        if status_code >= 400:
            self.error_count[status_code] += 1

    def record_inference(self, duration: float):
        """Record model inference metrics."""
        self.model_inference_time.observe(duration)
        self.model_requests.inc()

    def get_stats(self) -> Dict:
        """Get current monitoring statistics."""
        if not self.request_times:
            return {"message": "No requests recorded yet"}
        
        avg_response_time = sum(self.request_times) / len(self.request_times)
        
        stats = {
            "total_requests": len(self.request_times),
            "average_response_time": avg_response_time,
            "error_count": dict(self.error_count),
            "endpoint_stats": dict(self.endpoint_stats)
        }
        
        return stats

# Global monitor instance
monitor = APIMonitor()

class MonitoringMiddleware:
    """FastAPI middleware for monitoring."""
    
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration = time.time() - start_time
                monitor.record_request(
                    method=scope["method"],
                    endpoint=scope["path"],
                    status_code=message["status"],
                    duration=duration
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)

