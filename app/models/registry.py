"""KGK AI Model Registry — Provider registration and selection.

Allows registering multiple model providers and selecting the active one
by name. This enables runtime model swapping without code changes.
"""

from __future__ import annotations

from typing import Optional

from app.models.base import BaseModelProvider, ModelInfo


class ModelRegistry:
    """Registry for model providers.

    Providers register themselves with a name. The active provider
    is used for all generation calls.
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseModelProvider]] = {}
        self._active_provider: Optional[BaseModelProvider] = None
        self._active_name: str = ""

    def register(
        self, name: str, provider_class: type[BaseModelProvider]
    ) -> None:
        """Register a model provider class.

        Args:
            name: Unique provider name (e.g. 'local_transformers').
            provider_class: Class implementing BaseModelProvider.
        """
        self._providers[name] = provider_class

    def get_provider_class(self, name: str) -> Optional[type[BaseModelProvider]]:
        """Retrieve a registered provider class by name."""
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def set_active(self, provider: BaseModelProvider, name: str = "") -> None:
        """Set the active provider instance.

        Args:
            provider: An instantiated BaseModelProvider.
            name: Optional name for the provider.
        """
        self._active_provider = provider
        self._active_name = name or provider.__class__.__name__

    @property
    def active(self) -> Optional[BaseModelProvider]:
        """Return the currently active provider, or None."""
        return self._active_provider

    @property
    def active_name(self) -> str:
        """Return the name of the active provider."""
        return self._active_name

    def is_ready(self) -> bool:
        """Check if a provider is active and healthy."""
        return self._active_provider is not None and self._active_provider.health_check()

    def get_info(self) -> Optional[ModelInfo]:
        """Return info about the active provider."""
        if self._active_provider is None:
            return None
        return self._active_provider.get_info()


registry = ModelRegistry()
