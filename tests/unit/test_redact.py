import random
import string

import pytest

from cvpilot.core.redact import REDACTED, redact_argv, redact_command_line, redact_mapping


@pytest.mark.parametrize(
    ("line", "secret"),
    [
        ("train.py --token=abc123", "abc123"),
        ("train.py --token abc123", "abc123"),
        ("train.py --password hunter2", "hunter2"),
        ("train.py --password=hunter2", "hunter2"),
        ("train.py --api-key sk-abc", "sk-abc"),
        ("train.py --api_key=sk-abc", "sk-abc"),
        ("train.py --hf-token abc123", "abc123"),
        ("train.py --jwt eyJhbGciOi", "eyJhbGciOi"),
        ("train.py --authorization 'Bearer abc123'", "abc123"),
        ('train.py --token="abc 123"', "abc 123"),
        ("train.py --TOKEN=abc123", "abc123"),
        ("WANDB_API_KEY=abc123 train.py", "abc123"),
        ("HF_TOKEN=abc123 python train.py", "abc123"),
        ("curl -H 'Authorization: Bearer abc123' x", "abc123"),
        ("train.py --db https://user:hunter2@host/db", "hunter2"),
    ],
)
def test_command_line_secrets_are_removed(line: str, secret: str) -> None:
    out = redact_command_line(line)
    assert secret not in out
    assert REDACTED in out


def test_non_secret_flags_survive_unchanged() -> None:
    line = "train.py --epochs 10 --lr=0.01 --token-count 5 --passwords-file x --data /d"
    assert redact_command_line(line) == line


def test_flag_value_that_is_another_flag_is_kept() -> None:
    out = redact_command_line("train.py --token --epochs 5")
    assert out == "train.py --token --epochs 5"


def test_redaction_is_idempotent() -> None:
    once = redact_command_line("train.py --token=abc --password x")
    assert redact_command_line(once) == once


def test_mapping_redacts_by_key_recursively() -> None:
    cfg = {
        "lr": 0.1,
        "api_key": "k",
        "nested": {"HF-Token": "t", "list": [{"password": "p"}, 3]},
        "pairs": [("jwt", "j"), ("ok", "v")],
        "Authorization": "Bearer x",
    }
    out = redact_mapping(cfg)
    assert out["lr"] == 0.1
    assert out["api_key"] == REDACTED
    assert out["Authorization"] == REDACTED
    assert out["nested"]["HF-Token"] == REDACTED
    assert out["nested"]["list"][0]["password"] == REDACTED
    assert out["nested"]["list"][1] == 3
    assert out["pairs"][0] == ["jwt", REDACTED]
    assert out["pairs"][1] == ["ok", "v"]
    assert cfg["api_key"] == "k"  # input is not mutated


def test_mapping_handles_cycles_and_depth() -> None:
    cyc: dict[str, object] = {}
    cyc["self"] = cyc
    out = redact_mapping(cyc)
    assert out["self"] == "[TRUNCATED]"


# --- review findings -------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "secret"),
    [
        ("PGPASSWORD=hunter2 psql", "hunter2"),
        ("AWS_SECRET_ACCESS_KEY=x1y2z3 train.py", "x1y2z3"),
        ("aws_secret_access_key=x1y2z3 train.py", "x1y2z3"),
        ("GITHUBTOKEN=x1y2z3 train.py", "x1y2z3"),
        ("train.py --accessToken x1y2z3", "x1y2z3"),
        ("train.py --clientSecret=x1y2z3", "x1y2z3"),
        ("train.py --apiKey x1y2z3", "x1y2z3"),
        ("PRIVATE_KEY=x1y2z3 train.py", "x1y2z3"),
        ("train.py --passwd x1y2z3", "x1y2z3"),
        ("train.py --pwd x1y2z3", "x1y2z3"),
        ("train.py --credentials x1y2z3", "x1y2z3"),
        ("train.py --auth x1y2z3", "x1y2z3"),
        ("train.py --cookie x1y2z3", "x1y2z3"),
        ("train.py --session-id x1y2z3", "x1y2z3"),
        ("train.py --access-key x1y2z3", "x1y2z3"),
        ("train.py --secret-key x1y2z3", "x1y2z3"),
        ("train.py --signing-key x1y2z3", "x1y2z3"),
        ("train.py --encryption_key x1y2z3", "x1y2z3"),
        (r'train.py --token "ab\"cd x1y2z3"', "x1y2z3"),
        (r"train.py --token 'ab\'cd x1y2z3'", "x1y2z3"),
        ('train.py --token "unterminated x1y2z3', "x1y2z3"),
        ("git clone https://ghp_abc123@github.com/x/y", "ghp_abc123"),
        ("train.py --url https://host/p?access_token=x1y2z3&a=1", "x1y2z3"),
        ("train.py --url https://host/p?a=1&sig=x1y2z3", "x1y2z3"),
        ("train.py --url https://host/p?X-Amz-Signature=x1y2z3", "x1y2z3"),
        ("curl -H 'Cookie: x1y2z3' x", "x1y2z3"),
    ],
)
def test_review_bypasses_are_closed(line: str, secret: str) -> None:
    out = redact_command_line(line)
    assert secret not in out and REDACTED in out
    assert redact_command_line(out) == out


@pytest.mark.parametrize(
    "line",
    [
        "train.py --token_count 5 --max_tokens 10 --author me --keyboard x --design=y",
        "train.py --tokenizer bert --passwords-file f --authority a --keys 3 --monkey=1",
        "git clone https://github.com/x/y --depth 1",
        "train.py --url https://host/p?a=1&b=2",
    ],
)
def test_innocent_flags_are_not_redacted(line: str) -> None:
    assert redact_command_line(line) == line


@pytest.mark.parametrize(
    "key",
    [
        "token", "password", "api_key", "apiKey", "ApiKey", "accessToken", "access_token",
        "clientSecret", "PRIVATE_KEY", "secret_key", "passwd", "pwd", "credentials", "auth",
        "cookie", "session_id", "sessionId", "bearer", "signing_key", "encryptionKey",
        "GITHUBTOKEN", "HF-Token", "x-api-key", "Authorization", "jwt", "DB_PASSWORD",
    ],
)  # fmt: skip
def test_sensitive_keys(key: str) -> None:
    assert redact_mapping({key: "v"})[key] == REDACTED


@pytest.mark.parametrize(
    "key",
    ["token_count", "max_tokens", "author", "keyboard", "key", "tokenizer", "monkey", "lr", "keys"],
)
def test_innocent_keys_are_kept(key: str) -> None:
    assert redact_mapping({key: "v"})[key] == "v"


def test_bytes_and_objects_in_config_never_leak() -> None:
    class Creds:
        def __repr__(self) -> str:
            return "Creds(token=x1y2z3, user=bob)"

    out = redact_mapping({"blob": b"x1y2z3", "obj": Creds(), "ba": bytearray(b"x1y2z3")})
    assert out["blob"] == REDACTED and out["ba"] == REDACTED
    assert "x1y2z3" not in out["obj"] and isinstance(out["obj"], str)


def test_long_object_repr_is_truncated() -> None:
    class Big:
        def __repr__(self) -> str:
            return "x" * 5000

    assert len(redact_mapping({"o": Big()})["o"]) <= 210


def test_string_values_are_pattern_redacted() -> None:
    out = redact_mapping({"cmd": "run --token=x1y2z3", "n": 3, "f": 0.5, "b": True, "z": None})
    assert "x1y2z3" not in out["cmd"]
    assert (out["n"], out["f"], out["b"], out["z"]) == (3, 0.5, True, None)


def test_colliding_non_string_keys_keep_the_redacted_entry() -> None:
    out = redact_mapping({1: "a", "1": "b"})  # type: ignore[dict-item]
    assert len(out) == 1
    out = redact_mapping({"password": "a", b"password": "b"})  # type: ignore[dict-item]
    assert out == {"password": REDACTED}

    class K:
        def __str__(self) -> str:
            return "token"

    out = redact_mapping({K(): "secret-a", "token": "secret-b"})  # type: ignore[dict-item]
    assert out == {"token": REDACTED}


def test_mapping_redaction_is_idempotent() -> None:
    cfg = {"a": {"token": "x", "cmd": "--password=p", "l": [("jwt", "j"), 1]}, "n": 2}
    once = redact_mapping(cfg)
    assert redact_mapping(once) == once


# --- structural redaction of an argument list -------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--token=a b"], ["--token=" + REDACTED]),
        (["--token", "a b"], ["--token", REDACTED]),
        (["--token", 'a\\"b'], ["--token", REDACTED]),
        (["-p", "hunter2"], ["-p", REDACTED]),
        (["PGPASSWORD=x", "cmd"], ["PGPASSWORD=" + REDACTED, "cmd"]),
        (["cmd", "--api-key"], ["cmd", "--api-key"]),
        (["cmd", "--password", "-v"], ["cmd", "--password", "-v"]),
        (["https://user@host/path"], [f"https://{REDACTED}@host/path"]),
        (["https://user:pw@host"], [f"https://user:{REDACTED}@host"]),
        (
            ["--url", "https://h/?access_token=abc&x=1"],
            ["--url", f"https://h/?access_token={REDACTED}&x=1"],
        ),
        (
            ["--clientSecret=s3", "--apiKey", "k"],
            ["--clientSecret=" + REDACTED, "--apiKey", REDACTED],
        ),
        (["Authorization: Bearer abc"], [f"Authorization: {REDACTED}"]),
        (["Bearer abc"], [f"Bearer {REDACTED}"]),
        (["--header", "Cookie: sid=1"], ["--header", f"Cookie: {REDACTED}"]),
        (["--name=my run", "--lr=0.1", "x=1"], ["--name=my run", "--lr=0.1", "x=1"]),
        (["--token_count=5", "--max_tokens", "9"], ["--token_count=5", "--max_tokens", "9"]),
        (["GITHUBTOKEN=abc"], ["GITHUBTOKEN=" + REDACTED]),
        (["--token", "--epochs", "3"], ["--token", "--epochs", "3"]),
        (["--token", "--password", "x"], ["--token", "--password", REDACTED]),
        ([], []),
    ],
)
def test_redact_argv_is_structural(argv: list[str], expected: list[str]) -> None:
    assert redact_argv(argv) == expected


def test_redact_argv_non_ascii_and_newlines_inside_a_value() -> None:
    out = redact_argv(["--token=пароль\nline2 \u2603", "--note", "多言語\nok"])
    assert out == ["--token=" + REDACTED, "--note", "多言語\nok"]
    assert redact_argv(["--password", "pä\nss"]) == ["--password", REDACTED]


def test_redact_argv_executable_is_checked_too() -> None:
    assert redact_argv(["https://tok@host/run.sh", "--x"]) == [
        f"https://{REDACTED}@host/run.sh",
        "--x",
    ]
    assert redact_argv(["Bearer abc"])[0] == f"Bearer {REDACTED}"


def test_redact_argv_does_not_mutate_and_is_idempotent() -> None:
    argv = ["t.py", "--token", "a b", "PGPASSWORD=x", "https://u:p@h/?sig=z", "-p", "q"]
    snapshot = list(argv)
    once = redact_argv(argv)
    assert argv == snapshot
    assert redact_argv(once) == once


def test_free_text_inside_a_whitespace_token_is_also_scanned() -> None:
    out = redact_argv(["--cmd", "run --token=abc123 --lr 1"])
    assert "abc123" not in " ".join(out)


def test_short_secret_flag_set_is_documented() -> None:
    from cvpilot.core import redact

    assert redact.SHORT_SECRET_FLAGS == frozenset({"-p"})
    assert "-p" in (redact.__doc__ or "")


def test_randomised_secrets_never_survive_redact_argv() -> None:
    rng = random.Random(1234)  # noqa: S311  # deterministic test data, not security
    word = string.ascii_letters + string.digits
    rich = word + " \"'\\=&?:;$`\n\t" + "пароль多"
    forms_free = [
        lambda s: ["--token=" + s],
        lambda s: ["--token", s],
        lambda s: ["--api-key", s],
        lambda s: ["--hfToken=" + s],
        lambda s: ["MY_SECRET=" + s],
        lambda s: ["PGPASSWORD=" + s],
        lambda s: ["-p", s],
        lambda s: ["--password=" + s],
        lambda s: ["Authorization: Bearer " + s],
        lambda s: ["--header", "Cookie: " + s],
    ]
    forms_url = [
        lambda s: [f"https://{s}@host/x"],
        lambda s: [f"https://user:{s}@host/x"],
        lambda s: [f"https://h/p?a=1&access_token={s}&b=2"],
        lambda s: ["--url", f"https://h/p?sig={s}"],
        lambda s: ["Bearer " + s],
    ]
    for _ in range(400):
        decoys = ["train.py", "--epochs", str(rng.randint(1, 9)), "--name=run-" + str(rng.random())]
        if rng.random() < 0.5:
            secret = "S" + "".join(rng.choice(rich) for _ in range(rng.randint(8, 20))) + "Z"
            tokens = rng.choice(forms_free)(secret)
        else:
            secret = "S" + "".join(rng.choice(word) for _ in range(rng.randint(8, 20))) + "Z"
            tokens = rng.choice(forms_url)(secret)
        argv = decoys[:2] + tokens + decoys[2:]
        out = redact_argv(argv)
        assert secret not in "\x00".join(out), (argv, out)
        assert redact_argv(out) == out
