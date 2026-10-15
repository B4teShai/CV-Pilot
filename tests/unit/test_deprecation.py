"""The deprecation helper and the policy it enforces."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

import cvpilot
from cvpilot import deprecation
from cvpilot.deprecation import CvpilotDeprecationWarning, deprecated, warn_deprecated


def test_a_deprecated_function_warns_with_everything_a_user_needs() -> None:
    @deprecated(since="0.2.0", removal="0.4.0", alternative="new_thing")
    def old(a: int, *, b: int = 2) -> int:
        """Add."""
        return a + b

    with pytest.warns(CvpilotDeprecationWarning) as caught:
        assert old(1, b=3) == 4
    text = str(caught[0].message)
    assert "old" in text and "0.2.0" in text and "0.4.0" in text and "new_thing" in text
    assert caught[0].filename == __file__  # points at the caller, not at the helper


def test_the_wrapper_keeps_name_signature_and_notes_the_deprecation() -> None:
    @deprecated(since="0.2.0", removal="0.4.0")
    def old(a: int, b: int = 2) -> int:
        """Add two numbers."""
        return a + b

    original = old.__wrapped__  # type: ignore[attr-defined]
    assert old.__name__ == "old"
    assert inspect.signature(old) == inspect.signature(original)
    assert old.__doc__ is not None and "Deprecated since 0.2.0" in old.__doc__
    assert "Add two numbers." in old.__doc__


def test_a_deprecated_method_and_the_class_registry() -> None:
    class Thing:
        @deprecated(since="0.2.0", removal="0.5.0", alternative="Thing.better")
        def old(self) -> str:
            return "x"

    with pytest.warns(CvpilotDeprecationWarning):
        assert Thing().old() == "x"
    assert any("Thing.old" in entry.name for entry in deprecation.registered())


@pytest.mark.parametrize(
    ("since", "removal"),
    [("0.2.0", "0.2.0"), ("0.3.0", "0.2.0"), ("x", "0.3.0"), ("0.2", "0.3.0"), ("0.2.0", "1")],
)
def test_the_removal_must_come_after_a_valid_since(since: str, removal: str) -> None:
    with pytest.raises(ValueError):
        deprecated(since=since, removal=removal)


def test_the_policy_requires_at_least_one_release_of_warning() -> None:
    """Before 1.0 a minor version may break the API, but only after a release that warned."""
    with pytest.raises(ValueError, match="at least one"):
        deprecated(since="0.2.0", removal="0.2.1")
    deprecated(since="0.2.0", removal="0.3.0")


def test_warn_deprecated_is_available_for_parameters_and_modules() -> None:
    with pytest.warns(CvpilotDeprecationWarning, match="parameter 'x'"):
        warn_deprecated("parameter 'x'", since="0.2.0", removal="0.4.0", alternative="'y'")


def test_the_warning_is_shown_by_default_to_end_users() -> None:
    """DeprecationWarning is hidden outside ``__main__``; this subclass is not."""
    assert issubclass(CvpilotDeprecationWarning, DeprecationWarning)
    code = (
        "import warnings, cvpilot.deprecation as d;"
        "print([f[0] for f in warnings.filters if f[2] is d.CvpilotDeprecationWarning])"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "['default']"


def test_nothing_registered_is_due_for_removal_in_this_version() -> None:
    expired = [e for e in deprecation.expired(cvpilot.__version__) if e.name.startswith("cvpilot.")]
    assert expired == [], f"remove these before releasing {cvpilot.__version__}: {expired}"


def test_expired_compares_versions_numerically() -> None:
    @deprecated(since="0.9.0", removal="0.10.0")
    def old_numeric() -> None:
        return None

    names = [e.name for e in deprecation.expired("0.10.0")]
    assert any(n.endswith("old") for n in names)
    assert not [e for e in deprecation.expired("0.9.5") if e.name.endswith("old_numeric")]
