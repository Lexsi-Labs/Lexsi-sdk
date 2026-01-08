from lexsi_sdk.client.client import APIClient
from lexsi_sdk.common.environment import Environment
from lexsi_sdk.core.xai import XAI


class LEXSI(XAI):
    """Base entry-point class for interacting with the Lexsi.ai platform. Handles authentication, organization discovery and selection, notification retrieval, and provides access to higher-level SDK abstractions."""
    pass
    # env: Environment = Environment()
    # api_client: APIClient = APIClient()