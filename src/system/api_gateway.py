"""
API Gateway — Step 2: System Design (Enterprise)

AIPulse Enterprise AI Platform — API Gateway Layer
Manages all incoming requests, authentication, rate limiting, and routing.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/api_gateway.log"),
    ],
)
logger = logging.getLogger("aipulse.api_gateway")


class RateLimiter:
    """Rate limiter for API endpoints."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, user_id: str, endpoint: str) -> bool:
        """Check if request is allowed."""
        key = f"{user_id}:{endpoint}"
        now = time.time()

        # Clean old entries
        self.requests[key] = [
            t for t in self.requests.get(key, [])
            if now - t < self.window_seconds
        ]

        if len(self.requests[key]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {key}")
            return False

        self.requests[key].append(now)
        return True

    def get_usage(self, user_id: str, endpoint: str) -> Dict:
        """Get current usage."""
        key = f"{user_id}:{endpoint}"
        return {
            "endpoint": endpoint,
            "user_id": user_id,
            "remaining": self.max_requests - len(self.requests.get(key, [])),
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
        }


class APIKeyManager:
    """Manage API keys for authentication."""

    def __init__(self, key_file: str = "config/api_keys.json"):
        self.key_file = key_file
        self.keys = self._load_keys()

    def _load_keys(self) -> Dict[str, str]:
        """Load API keys from file."""
        if os.path.exists(self.key_file):
            with open(self.key_file, "r") as f:
                return json.load(f)
        return {"ai_model": "your-api-key-here"}

    def validate(self, api_key: str, expected_key: str) -> bool:
        """Validate API key."""
        return api_key == self.keys.get(expected_key, "")

    def get_key(self, endpoint: str) -> str:
        """Get API key for endpoint."""
        return self.keys.get(endpoint, "")


class RequestRouter:
    """Route incoming requests to appropriate handlers."""

    def __init__(self, handlers: Dict[str, Any]):
        self.handlers = handlers

    def route(self, method: str, path: str, payload: Dict) -> Dict:
        """Route request to appropriate handler."""
        handler_key = f"{method}:{path}"

        if handler_key not in self.handlers:
            return {
                "status": 404,
                "message": f"Endpoint not found: {method} {path}",
            }

        handler = self.handlers[handler_key]
        logger.info(f"Routing request to {handler_key}")
        return handler(payload)


class RequestHandler:
    """Handle individual API requests."""

    def __init__(self, route: str, handler_func):
        self.route = route
        self.handler_func = handler_func

    def __call__(self, payload: Dict) -> Dict:
        """Execute handler function."""
        return self.handler_func(payload)


class APIGateway:
    """Main API Gateway for AIPulse."""

    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        self.api_key_manager = APIKeyManager()
        self.router = RequestRouter({})
        self.monitoring = MonitoringModule()
        self.gateway_config = {
            "max_request_size": 10_485_760,  # 1MB
            "timeout": 30,  # seconds
            "max_retries": 3,
            "retry_delay": 1,
            "rate_limit": {"enabled": True, "max_requests": 100, "window_seconds": 60},
        }

    def handle_request(self, method: str, path: str, body: Optional[Dict] = None) -> Dict:
        """Handle incoming API request."""
        # Rate limiting
        user_id = self._get_user_id(body) if body else "anonymous"
        if not self.rate_limiter.is_allowed(user_id, path):
            return {
                "status": 429,
                "message": "Rate limit exceeded",
                "retry_after": self.gateway_config["rate_limit"]["window_seconds"],
            }

        # Logging
        logger.info(f"Request: {method} {path}")

        # Routing
        handler = self.router.router.get(f"{method}:{path}")
        if handler:
            return handler(body)

        return {
            "status": 404,
            "message": "Endpoint not found",
        }

    def _get_user_id(self, body: Dict) -> str:
        """Extract user ID from request body."""
        if body and "user_id" in body:
            return body["user_id"]
        return "anonymous"


# Entry point
def main():
    """Run the API Gateway."""
    gateway = APIGateway()

    # Test
    result = gateway.handle_request("GET", "/api/v1/health")
    print(f"Gateway Health Check: {result}")

    result = gateway.handle_request("POST", "/api/v1/models/train", {"model_type": "lstm"})
    print(f"Model Training: {result}")

    logger.info("API Gateway started successfully!")


if __name__ == "__main__":
    main()
