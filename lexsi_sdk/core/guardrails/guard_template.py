from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union


class Guard:
    """Collection of predefined guardrail templates for common safety, PII, toxicity, and compliance checks."""

    @staticmethod
    def detect_pii(entities: List[str] = None) -> Dict[str, Any]:
        """Create a guardrail configuration that detects personally identifiable information (PII) in text. Optionally specify a list of entities (e.g., EMAIL_ADDRESS, PHONE_NUMBER) to detect.

        :param entities: List of PII entity types to detect.
        """
        if entities is None:
            entities = ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "ADDRESS"]

        return {"name": "Detect PII", "config": {"pii_entities": entities}}

    @staticmethod
    def nsfw_text(
        threshold: float = 0.8, validation_method: str = "sentence"
    ) -> Dict[str, Any]:
        """Create a guardrail configuration that flags NSFW text content. Accepts a threshold value (0.0–1.0) to control sensitivity and a validation_method to specify the granularity (sentence, paragraph, document).

        :param threshold: Confidence threshold for detection (0.0-1.0).
        :param validation_method: Validation scope: "sentence", "paragraph", or "document".
        """
        return {
            "name": "NSFW Text",
            "config": {"threshold": threshold, "validation_method": validation_method},
        }

    @staticmethod
    def ban_list(banned_words: List[str]) -> Dict[str, Any]:
        """Create a guardrail configuration to detect banned words or phrases. Provide a list of banned_words to flag any occurrences."""
        return {"name": "Ban List", "config": {"banned_words": banned_words}}

    @staticmethod
    def bias_check(threshold: float = 0.9) -> Dict[str, Any]:
        """Create a guardrail configuration that checks content for bias. Accepts a threshold (default 0.9) to determine sensitivity to bias."""
        return {"name": "Bias Check", "config": {"threshold": threshold}}

    @staticmethod
    def competitor_check(competitors: List[str]) -> Dict[str, Any]:
        """Create a guardrail configuration that flags mentions of competitors. Provide a list of competitor names to detect."""
        return {"name": "Competitor Check", "config": {"competitors": competitors}}

    @staticmethod
    def correct_language(
        expected_language_iso: str = "en", threshold: float = 0.75
    ) -> Dict[str, Any]:
        """Template for correct language guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {
            "name": "Correct Language",
            "config": {
                "expected_language_iso": expected_language_iso,
                "threshold": threshold,
            },
        }

    @staticmethod
    def gibberish_text(
        threshold: float = 0.5, validation_method: str = "sentence"
    ) -> Dict[str, Any]:
        """Template for gibberish text guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {
            "name": "Gibberish Text",
            "config": {"threshold": threshold, "validation_method": validation_method},
        }

    @staticmethod
    def profanity_free() -> Dict[str, Any]:
        """Template for profanity free guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Profanity Free", "config": {}}

    @staticmethod
    def secrets_present() -> Dict[str, Any]:
        """Template for secrets present guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Secrets Present", "config": {}}

    @staticmethod
    def toxic_language(
        threshold: float = 0.5, validation_method: str = "sentence"
    ) -> Dict[str, Any]:
        """Template for toxic language guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {
            "name": "Toxic Language",
            "config": {"threshold": threshold, "validation_method": validation_method},
        }

    @staticmethod
    def contains_string(substring: str) -> Dict[str, Any]:
        """Template for contains string guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Contains String", "config": {"substring": substring}}

    @staticmethod
    def detect_jailbreak(threshold: float = 0.0) -> Dict[str, Any]:
        """Template for detect jailbreak guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Detect Jailbreak", "config": {"threshold": threshold}}

    @staticmethod
    def endpoint_is_reachable() -> Dict[str, Any]:
        """Template for endpoint is reachable guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Endpoint Is Reachable", "config": {}}

    @staticmethod
    def ends_with(end: str) -> Dict[str, Any]:
        """Template for ends with guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Ends With", "config": {"end": end}}

    @staticmethod
    def has_url() -> Dict[str, Any]:
        """Template for has url guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Has Url", "config": {}}

    @staticmethod
    def lower_case() -> Dict[str, Any]:
        """Template for lower case guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Lower Case", "config": {}}

    @staticmethod
    def mentions_drugs() -> Dict[str, Any]:
        """Template for mentions drugs guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Mentions Drugs", "config": {}}

    @staticmethod
    def one_line() -> Dict[str, Any]:
        """Template for one line guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "One Line", "config": {}}

    @staticmethod
    def reading_time(reading_time: float) -> Dict[str, Any]:
        """Template for reading time guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Reading Time", "config": {"reading_time": reading_time}}

    @staticmethod
    def redundant_sentences(threshold: int = 70) -> Dict[str, Any]:
        """Template for redundant sentences guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Redundant Sentences", "config": {"threshold": threshold}}

    @staticmethod
    def regex_match(regex: str, match_type: str = "search") -> Dict[str, Any]:
        """Template for regex match guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {
            "name": "Regex Match",
            "config": {"regex": regex, "match_type": match_type},
        }

    @staticmethod
    def sql_column_presence(cols: List[str]) -> Dict[str, Any]:
        """Template for SQL column presence guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Sql Column Presence", "config": {"cols": cols}}

    @staticmethod
    def two_words() -> Dict[str, Any]:
        """Template for two words guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Two Words", "config": {}}

    @staticmethod
    def upper_case() -> Dict[str, Any]:
        """Template for upper case guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Upper Case", "config": {}}

    @staticmethod
    def valid_choices(choices: List[str]) -> Dict[str, Any]:
        """Template for valid choices guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Valid Choices", "config": {"choices": choices}}

    @staticmethod
    def valid_json() -> Dict[str, Any]:
        """Template for valid json guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Valid Json", "config": {}}

    @staticmethod
    def valid_length(
        min: Optional[int] = None, max: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a guardrail configuration that checks whether the length of text falls within a specified range. Accepts optional min and/or max values."""
        config = {}
        if min is not None:
            config["min"] = min
        if max is not None:
            config["max"] = max
        return {"name": "Valid Length", "config": config}

    @staticmethod
    def valid_range(
        min: Optional[int] = None, max: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a guardrail configuration to validate that a numeric value is within a specified range. Accepts optional min and max values."""
        config = {}
        if min is not None:
            config["min"] = min
        if max is not None:
            config["max"] = max
        return {"name": "Valid Range", "config": config}

    @staticmethod
    def valid_url() -> Dict[str, Any]:
        """Create a guardrail configuration that checks if a string is a valid URL."""
        return {"name": "Valid URL", "config": {}}

    @staticmethod
    def web_sanitization() -> Dict[str, Any]:
        """Template for web sanitization guardrail
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return {"name": "Web Sanitization", "config": {}}
