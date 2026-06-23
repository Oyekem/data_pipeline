from datetime import datetime
from sqlalchemy import text

class PipelineMetrics:
    def __init__(self, engine):
        self.engine = engine

        self.total_runs = 0
        self.success_runs = 0
        self.failed_runs = 0
        self.last_run_time = None
        self.last_duration = 0

    def _save_to_db(self, status, duration):
        query = text("""
            INSERT INTO pipeline_metrics (
                total_runs,
                success_runs,
                failed_runs,
                duration_seconds,
                status
            )
            VALUES (
                :total_runs,
                :success_runs,
                :failed_runs,
                :duration_seconds,
                :status
            )
        """)

        with self.engine.begin() as conn:
            conn.execute(query, {
                "total_runs": self.total_runs,
                "success_runs": self.success_runs,
                "failed_runs": self.failed_runs,
                "duration_seconds": duration,
                "status": status
            })

    def record_success(self, duration):
        self.total_runs += 1
        self.success_runs += 1
        self.last_duration = duration
        self.last_run_time = datetime.now()

        self._save_to_db("SUCCESS", duration)

    def record_failure(self, duration):
        self.total_runs += 1
        self.failed_runs += 1
        self.last_duration = duration
        self.last_run_time = datetime.now()

        self._save_to_db("FAILED", duration)

    def health_status(self):
        if self.total_runs == 0:
            return "NO DATA"

        failure_rate = self.failed_runs / self.total_runs

        if failure_rate < 0.1:
            return "HEALTHY"
        elif failure_rate < 0.3:
            return "DEGRADED"
        else:
            return "FAILING"