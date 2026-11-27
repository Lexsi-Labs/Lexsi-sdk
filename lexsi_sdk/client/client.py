import atexit
import os
import httpx
import time as time_module
from lexsi_sdk.common.xai_uris import LOGIN_URI
import jwt
import threading
from typing import Optional
from pydantic import BaseModel, Field
import json

_TOKEN_REFRESH_LOCK = threading.Lock()
_HTTP_CLIENT_LOCK = threading.Lock()
_http_client: Optional[httpx.Client] = None
_http_client_pid: Optional[int] = None

_HTTP_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=30.0,
)


def get_http_client() -> httpx.Client:
    """Return the process-wide httpx client, creating it on first use.

    Lazy on purpose: importing the SDK must not open sockets or build a pool.
    """
    global _http_client, _http_client_pid
    pid = os.getpid()
    with _HTTP_CLIENT_LOCK:
        if _http_client is not None and _http_client_pid != pid:
            _http_client = None
        if _http_client is None:
            _http_client = httpx.Client(
                http2=True,
                timeout=None,
                limits=_HTTP_LIMITS,
            )
            _http_client_pid = pid
        return _http_client


def close_http_client() -> None:
    """Close the shared pool. It is recreated on the next request."""
    global _http_client, _http_client_pid
    with _HTTP_CLIENT_LOCK:
        client, _http_client = _http_client, None
        _http_client_pid = None
    if client is not None:
        client.close()


atexit.register(close_http_client)



class LexsiAPIError(Exception):
    """Raised when the API returns an HTTP error (4xx/5xx), a redirect, or a non-JSON reply.

    Attributes:
        status_code: HTTP status of the response.
        detail: parsed error message (backend 'details'/'data'/'detail' when present, else raw text).
        body: parsed JSON body when available, otherwise None.
    """

    def __init__(self, status_code, detail, body=None):
        self.status_code = status_code
        self.detail = detail
        self.body = body
        super().__init__(f"[{status_code}] {detail}")


class APIClient(BaseModel):
    """API client to interact with Lexsi.ai services"""

    debug: bool = False
    base_url: str = ""
    access_token: str = Field("", repr=False)
    auth_token: str = Field("", repr=False)
    headers: dict = Field(default={"Lexsi-Client": "Lexsi-SDK"}, repr=False)

    def __init__(self, **kwargs):
        """Initialize the API client with provided configuration.
        Stores configuration and prepares the object for use."""
        super().__init__(**kwargs)

    @property
    def client(self) -> httpx.Client:
        """The shared HTTP client used for all requests.

        Exposed as a property (not a pydantic field) because ``httpx.Client``
        holds locks and cannot be validated/serialised by pydantic.
        """
        return get_http_client()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        """Close the shared HTTP connection pool.

        The pool is recreated lazily on the next request, so this does not
        permanently break other clients in the process.
        """
        close_http_client()

    def get_auth_token(self) -> str:
        """get jwt auth token value

        Returns:
            str: jwt auth token
        """
        return self.auth_token

    def set_auth_token(self, auth_token):
        """sets jwt auth token value

        :param auth_token: jwt auth token
        """
        self.auth_token = auth_token

    def set_access_token(self, access_token):
        """sets access token value

        :param auth_token: jwt auth token
        """
        self.access_token = access_token

    def get_url(self, uri) -> str:
        """get url by appending uri to base url

        :param uri: uri of endpoint
        :return: url
        """
        return f"{self.base_url}/{uri}"

    def update_headers(self, auth_token):
        """sets jwt auth token and updates headers for all requests
        Encapsulates a small unit of SDK logic and returns the computed result."""
        self.set_auth_token(auth_token)
        self.headers = {
            **self.headers,
            "Authorization": f"Bearer {self.auth_token}",
        }

    def _token_expired(self) -> bool:
        """Cheap local check for token expiry (no network call).

        :raises jwt.exceptions.DecodeError: if the token is malformed
        :return: True when the token exists and is expired
        """
        if not self.auth_token:
            return False
        try:
            jwt.decode(
                self.auth_token,
                options={"verify_signature": False, "verify_exp": True},
            )
            return False
        except jwt.exceptions.ExpiredSignatureError:
            return True

    def refresh_bearer_token(self):
        """Refresh the bearer token if the current token is expired.

        Single-flight: concurrent callers that find the token expired share
        one login — the first takes the lock and logs in, the rest wait,
        re-check expiry and reuse the fresh token. The common unexpired
        path never takes the lock.
        """
        if not self._token_expired():
            return
        with _TOKEN_REFRESH_LOCK:
            if not self._token_expired():
                return
            response = self.base_request(
                "POST", LOGIN_URI, {"access_token": self.access_token}
            ).json()
            self.update_headers(response["access_token"])

    def base_request(self, method, uri, payload={}, files=None):
        """makes request to xai base service

        :param uri: api uri
        :param method: GET, POST, PUT, DELETE
        :raises LexsiAPIError: HTTP error (4xx/5xx), unexpected redirect, or non-JSON reply
        :return: JSON response
        """
        url = f"{self.base_url}/{uri}"
        try:
            request_kwargs = {
                "method": method,
                "url": url,
                "headers": self.headers,
                "files": files or None,
            }
            if payload or method.upper() not in ("GET", "HEAD"):
                request_kwargs["json"] = payload
            response = self.client.request(**request_kwargs)

            status = response.status_code
            if 300 <= status < 400:
                location = response.headers.get("location", "")
                raise LexsiAPIError(
                    status,
                    f"Unexpected redirect to {location or 'unknown location'}; expected a JSON response",
                )
            if status >= 400:
                try:
                    body = response.json()
                except Exception:
                    body = None
                if isinstance(body, dict):
                    detail = body.get("details") or body.get("data") or body.get("detail") or body
                elif body is not None:
                    detail = body
                else:
                    detail = response.text
                raise LexsiAPIError(status, detail, body=body)
            try:
                response.json()
            except Exception:
                text = response.text
                raise LexsiAPIError(
                    status,
                    f"Expected a JSON response, got non-JSON body: {text[:200] if text else 'empty body'}",
                )
            return response
        except Exception as e:
            raise e

    def request(self, method, uri, payload):
        """Refresh credentials and dispatch a base request.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        self.refresh_bearer_token()
        response = self.base_request(method, uri, payload)
        return response

    def get(self, uri):
        """makes get request to xai base service

        :param uri: api uri
        :raises Exception: Request exception
        :return: JSON response
        """

        self.refresh_bearer_token()
        response = self.base_request("GET", uri)
        return response.json()

    def post(self, uri, payload={}):
        """makes post request to xai base service

        :param uri: api uri
        :param payload: api payload, defaults to {}
        :raises Exception: Request exception
        :return: JSON response
        """

        self.refresh_bearer_token()
        response = self.base_request("POST", uri, payload)

        return response.json()
    def delete(self, uri, payload={}):
        """makes post request to xai base service

        :param uri: api uri
        :raises Exception: Request exception
        :return: JSON response
        """

        self.refresh_bearer_token()
        response = self.base_request("DELETE", uri)

        return response.json()
    def put(self, uri, payload={}):
        """makes post request to xai base service

        :param uri: api uri
        :raises Exception: Request exception
        :return: JSON response
        """

        self.refresh_bearer_token()
        response = self.base_request("put", uri , payload)

        return response.json()

    def stream(
        self,
        uri,
        method,
        payload=None,
        retry=False,
        max_consecutive_failures=8,
        initial_delay=2.0,
        max_delay=15.0,
    ):
        """Server-Sent Events / line-streaming endpoint.
        When ``retry`` is enabled, reconnect after transport failures with
        exponential backoff. Retries are opt-in so other streaming calls keep
        their existing behavior.
        """
        url = f"{self.base_url}/{uri}"

        # streaming MUST be consumed inside the context
        self.refresh_bearer_token()
        headers = {**self.headers, "Accept": "text/event-stream"}
        with self.client.stream(method, url, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                try:
                    error_body = response.read().decode()
                except Exception:
                    error_body = "Something went wrong while streaming response"
                raise LexsiAPIError(response.status_code, error_body)
            for line in response.iter_lines():  # no decode_unicode arg in httpx
                if not line:
                    continue
                if line.startswith("data: "):  # typical SSE prefix
                    if line.strip() == "data: [DONE]":
                        break
                    yield json.loads(line[6:])

        attempt = 0
        while True:
            try:
                self.refresh_bearer_token()
                headers = {**self.headers, "Accept": "text/event-stream"}
                with self.client.stream(method, url, headers=headers, json=payload) as response:
                    if response.status_code >= 400:
                        try:
                            error_body = response.read().decode()
                        except Exception:
                            error_body = "Something went wrong while streaming response"
                        raise Exception(error_body)
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            if line.strip() == "data: [DONE]":
                                break
                            event = json.loads(line[6:])
                            attempt = 0
                            yield event
                return
            except httpx.TransportError as exc:
                if not retry:
                    raise
                attempt += 1
                if attempt >= max_consecutive_failures:
                    raise
                delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                print(
                    f"Event stream connection lost "
                    f"({exc.__class__.__name__}: {exc}); reconnecting in "
                    f"{delay:.0f}s (attempt {attempt}/{max_consecutive_failures})..."
                )
                time_module.sleep(delay)

    def file(self, uri, files):
        """makes multipart request to send files

        :param uri: api uri
        :param file_path: file path
        :return: JSON response
        """
        self.refresh_bearer_token()
        response = self.base_request("POST", uri, files=files)
        return response.json()
