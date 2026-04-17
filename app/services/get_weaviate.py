import weaviate
from weaviate.auth import AuthApiKey
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

_client = None


def get_weaviate_client() -> weaviate.WeaviateClient:
    global _client
    if _client is None or not _client.is_connected():
        log.info("Connecting to Weaviate Cloud...")
        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.WEAVIATE_URL,
            auth_credentials=AuthApiKey(settings.WEAVIATE_API_KEY),
            skip_init_checks=True,
        )
        log.info("Weaviate Cloud connected.")
    return _client


def close_weaviate_client():
    global _client
    if _client and _client.is_connected():
        _client.close()
        log.info("Weaviate client closed.")
