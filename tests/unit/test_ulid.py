import re
import threading

import pytest

from cvpilot.core.hashing import new_run_id
from cvpilot.core.ulid import UlidGenerator, generate_ulid, is_valid_ulid, ulid_timestamp_ms

ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def test_format_is_26_crockford_characters() -> None:
    for _ in range(100):
        assert ULID_RE.match(generate_ulid())


def test_ids_are_strictly_monotonic_even_within_one_millisecond() -> None:
    ids = [generate_ulid() for _ in range(5000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_monotonic_with_a_frozen_clock() -> None:
    gen = UlidGenerator()
    ids = [gen.generate(now_ms=1_700_000_000_000) for _ in range(1000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 1000
    assert {ulid_timestamp_ms(i) for i in ids} == {1_700_000_000_000}


def test_clock_going_backwards_never_reorders_ids() -> None:
    gen = UlidGenerator()
    first = gen.generate(now_ms=1_700_000_000_500)
    second = gen.generate(now_ms=1_700_000_000_000)
    assert second > first


def test_timestamp_round_trips() -> None:
    assert (
        ulid_timestamp_ms(UlidGenerator().generate(now_ms=1_234_567_890_123)) == 1_234_567_890_123
    )


def test_thread_safe_uniqueness() -> None:
    out: list[str] = []
    lock = threading.Lock()

    def work() -> None:
        local = [generate_ulid() for _ in range(500)]
        with lock:
            out.extend(local)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(out)) == 4000


@pytest.mark.parametrize(
    "value",
    [
        "",
        "01ARZ3NDEKTSV4RRFFQ69G5FA",  # 25 chars
        "01ARZ3NDEKTSV4RRFFQ69G5FAVX",  # 27 chars
        "01ARZ3NDEKTSV4RRFFQ69G5FAI",  # I is not Crockford
        "01ARZ3NDEKTSV4RRFFQ69G5FAL",
        "01ARZ3NDEKTSV4RRFFQ69G5FAO",
        "01ARZ3NDEKTSV4RRFFQ69G5FAU",
        "01arz3ndektsv4rrffq69g5fav",  # lower case
        "81ARZ3NDEKTSV4RRFFQ69G5FAV",  # overflows 128 bits
        "../../etc/passwd0000000000",
    ],
)
def test_rejects_invalid_ids(value: str) -> None:
    assert not is_valid_ulid(value)


def test_accepts_spec_example() -> None:
    assert is_valid_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV")


def test_new_run_id_is_a_valid_ulid() -> None:
    assert is_valid_ulid(new_run_id())


def test_random_part_overflow_rolls_the_timestamp_forward() -> None:
    gen = UlidGenerator()
    first = gen.generate(now_ms=1_700_000_000_000)
    gen._last_rand = (1 << 80) - 1
    second = gen.generate(now_ms=1_700_000_000_000)
    assert second > first and ulid_timestamp_ms(second) == 1_700_000_000_001
