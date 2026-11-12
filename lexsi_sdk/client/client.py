import requests
import httpx
from lexsi_sdk.common.xai_uris import LOGIN_URI
import jwt
from pydantic import BaseModel
import json


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
    access_token: str = ""
    auth_token: str = ""
    headers: dict = {
        "Lexsi-Client": "Lexsi-SDK"
    }

    def __init__(self, **kwargs):
        """Initialize the API client with provided configuration.
        Stores configuration and prepares the object for use."""
        super().__init__(**kwargs)

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

    def refresh_bearer_token(self):
        """Refresh the bearer token if the current token is expired.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        try:
            if self.auth_token:
                jwt.decode(
                    self.auth_token,
                    options={"verify_signature": False, "verify_exp": True},
                )
        except jwt.exceptions.ExpiredSignatureError as e:
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
            # response = requests.request(
            #     method,
            #     url,
            #     headers=self.headers,
            #     json=payload,
            #     files=files,
            #     stream=stream,
            # )

            with httpx.Client(http2=True, timeout=None) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=payload,
                    files=files or None,
                )
                # response.raise_for_status()
                # return response

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

    def stream(self, uri, method, payload=None):
        """Server-Sent Events / line-streaming endpoint.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        self.refresh_bearer_token()
        url = f"{self.base_url}/{uri}"
        # if SSE, this header helps
        headers = {**self.headers, "Accept": "text/event-stream"}

        with httpx.Client(http2=True, timeout=None) as client:
            # streaming MUST be consumed inside the context
            with client.stream(method, url, headers=headers, json=payload) as response:
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

    def file(self, uri, files):
        """makes multipart request to send files

        :param uri: api uri
        :param file_path: file path
        :return: JSON response
        """
        self.refresh_bearer_token()
        response = self.base_request("POST", uri, files=files)
        return response.json()
