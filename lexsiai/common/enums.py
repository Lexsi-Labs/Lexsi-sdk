from enum import Enum


class UserRole(Enum):
    """Enumeration of supported user roles within Lexsi."""

    ADMIN = 'admin'
    USER = 'user'
