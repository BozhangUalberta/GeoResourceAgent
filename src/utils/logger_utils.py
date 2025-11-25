import logging
import json
import asyncio
from fastapi import WebSocket

# Global dictionary to store loggers by user ID
logger_registry = {}


class WebSocketHandler(logging.Handler):
    """Custom logging handler to send logs via WebSocket."""

    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket
        self.queue = asyncio.Queue()  # Create an async queue for log messages
        self.task = asyncio.create_task(self._process_logs())  # Start the queue processor

    def emit(self, record):
        """Add the log entry to the queue."""
        try:
            log_entry = self.format(record)
            self.queue.put_nowait(log_entry)  # Add log to the queue
        except Exception as e:
            print(f"Failed to enqueue log: {e}")

    async def _process_logs(self):
        """Process log messages asynchronously from the queue."""
        try:
            while True:
                log_entry = await self.queue.get()  # Wait for a log entry
                message = {
                    "event_type": "log",
                    "content": log_entry,
                }
                await self.websocket.send_text(json.dumps(message))  # Send log via WebSocket
        except Exception as e:
            print(f"Error processing logs: {e}")

    def close(self):
        """Cleanup the handler by canceling the async task."""
        self.task.cancel()
        super().close()



class UserIDFilter(logging.Filter):
    """A logging filter to add user_id to log records."""
    def __init__(self, user_id: str):
        super().__init__()
        self.user_id = user_id

    def filter(self, record):
        record.user_id = self.user_id
        return True


def get_logger(user_id: str):
    if user_id not in logger_registry:
        logger = logging.getLogger(f"{user_id}")
        logger.setLevel(logging.DEBUG)

        # Add a filter to include user_id in log records
        user_id_filter = UserIDFilter(user_id)
        logger.addFilter(user_id_filter)

        # Add a default console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [User-%(user_id)s]: %(message)s'
            ))
            logger.addHandler(console_handler)

        logger_registry[user_id] = logger

    return logger_registry[user_id]

def attach_ws_handler(logger: logging.Logger, websocket: WebSocket):
    """Attach a WebSocket handler to the logger."""
    ws_handler = WebSocketHandler(websocket)
    # ws_handler.setFormatter(logging.Formatter(
    #     '[%(asctime)s] [%(levelname)s]: %(message)s'
    # ))
    logger.addHandler(ws_handler)
    return ws_handler  # Return handler for removal later


def detach_ws_handler(logger: logging.Logger, handler: WebSocketHandler):
    """Remove the WebSocket handler from the logger."""
    logger.removeHandler(handler)