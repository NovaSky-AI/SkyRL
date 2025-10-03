"""Background engine for processing training requests."""
import time
import logging
from datetime import datetime, timezone
from sqlmodel import create_engine, Session, select

from tx.tinker.models import FutureDB, DB_PATH, RequestType, RequestStatus

logger = logging.getLogger(__name__)


class TinkerEngine:
    """Background engine for processing training requests."""

    def __init__(self, db_path=DB_PATH):
        """Initialize the engine with a database connection."""
        self.db_engine = create_engine(f"sqlite:///{db_path}", echo=False)

    def process_forward_backward(self, request_id: str, model_id: str, request_data: dict) -> dict:
        """Process a forward_backward request and return mock results."""
        # Mock implementation - returns dummy loss
        result = {
            "loss_fn_output_type": "scalar",
            "loss_fn_outputs": [{
                "loss": {
                    "data": [0.5],
                    "dtype": "float32",
                    "shape": [1]
                }
            }],
            "metrics": {}
        }
        return result

    def process_optim_step(self, request_id: str, model_id: str, request_data: dict) -> dict:
        """Process an optim_step request and return empty result."""
        # Mock implementation - returns empty dict
        return {}

    def process_pending_requests(self):
        """Main loop to process pending requests."""
        while True:
            with Session(self.db_engine) as session:
                # Get all pending requests
                statement = select(FutureDB).where(FutureDB.status == RequestStatus.PENDING)
                pending = session.exec(statement).all()

                for future in pending:
                    try:
                        # Process based on request type
                        if future.request_type == RequestType.FORWARD_BACKWARD:
                            result_data = self.process_forward_backward(
                                future.request_id,
                                future.model_id,
                                future.request_data
                            )
                        elif future.request_type == RequestType.OPTIM_STEP:
                            result_data = self.process_optim_step(
                                future.request_id,
                                future.model_id,
                                future.request_data
                            )
                        else:
                            logger.warning(f"Unknown request type: {future.request_type}")
                            continue

                        # Update the future with results
                        future.result_data = result_data
                        future.status = RequestStatus.COMPLETED
                        future.completed_at = datetime.now(timezone.utc)
                        session.add(future)
                        session.commit()

                        logger.info(f"Completed {future.request_type} request {future.request_id}")

                    except Exception as e:
                        logger.error(f"Error processing request {future.request_id}: {e}")
                        future.result_data = {"error": str(e)}
                        future.status = RequestStatus.FAILED
                        future.completed_at = datetime.now(timezone.utc)
                        session.add(future)
                        session.commit()

            # Poll every 100ms
            time.sleep(0.1)

    def run(self):
        """Entry point to start the engine."""
        logger.info("Starting background engine...")
        self.process_pending_requests()


def main():
    """Entry point for the background engine."""
    logging.basicConfig(level=logging.INFO)
    TinkerEngine().run()


if __name__ == "__main__":
    main()
