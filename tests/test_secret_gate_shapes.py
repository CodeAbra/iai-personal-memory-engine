"""Table-driven proof that the secret-shaped-content gate recognizes real
credential SHAPES, not just trigger words. One case per shape; PEM/OpenSSH
private-key blocks and bare JWTs are the highest-value cases -- the shapes
a trigger-word-only alphabet misses entirely.
"""

from __future__ import annotations

import pytest

from iai_mcp.tools import secret_gate

_BEGIN = "-----" + "BEGIN "
_END = "-----" + "END "
_PK = " PRIVATE " + "KEY-----\n"

PEM_RSA = (
    _BEGIN + "RSA" + _PK
    + "MIIEowIBAAKCAQEA1234567890abcdefghijklmnopqrstuvwxyz\n"
    + _END + "RSA" + _PK
)
PEM_PLAIN = (
    _BEGIN + _PK.lstrip()
    + "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEA\n"
    + _END + _PK.lstrip()
)
PEM_OPENSSH = (
    _BEGIN + "OPENSSH" + _PK
    + "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2g\n"
    + _END + "OPENSSH" + _PK
)
PEM_EC = (
    _BEGIN + "EC" + _PK
    + "MHcCAQEEIBaVvaryOU3JkQjt7WMDeeMHXhTOfz+wpFhwZo1sJ8fZoAoGCCqGSM49\n"
    + _END + "EC" + _PK
)
AWS_ACCESS_KEY_ID = "AKIA" + "ABCDEFGHIJKLMNOP"
GITHUB_PAT_CLASSIC = "ghp_" + "a1B2c3D4" * 5
GITHUB_PAT_FINE_GRAINED = "github_pat_" + "a1B2c3D4" * 6
SLACK_BOT_TOKEN = "xoxb-" + "1234567890-abcdefghijklmnop"
STRIPE_LIVE_KEY = "sk_live_" + "51H8xzyz1234567890abcd"
STRIPE_TEST_KEY = "sk_test_" + "51H8xzyz1234567890abcd"
GOOGLE_API_KEY = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"[:35]
SENTRY_TOKEN = "sntryu_" + "a1B2c3D4" * 5
BARE_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6ImFsaWNlIn0."
    "dQw4w9WgXcQ_fixture_signature_segment"
)

SECRET_SHAPES = [
    pytest.param(PEM_RSA, id="pem_rsa_private_key"),
    pytest.param(PEM_PLAIN, id="pem_plain_private_key"),
    pytest.param(PEM_OPENSSH, id="pem_openssh_private_key"),
    pytest.param(PEM_EC, id="pem_ec_private_key"),
    pytest.param(AWS_ACCESS_KEY_ID, id="aws_access_key_id"),
    pytest.param(GITHUB_PAT_CLASSIC, id="github_pat_classic"),
    pytest.param(GITHUB_PAT_FINE_GRAINED, id="github_pat_fine_grained"),
    pytest.param(SLACK_BOT_TOKEN, id="slack_bot_token"),
    pytest.param(STRIPE_LIVE_KEY, id="stripe_live_key"),
    pytest.param(STRIPE_TEST_KEY, id="stripe_test_key"),
    pytest.param(GOOGLE_API_KEY, id="google_api_key"),
    pytest.param(SENTRY_TOKEN, id="sentry_token"),
    pytest.param(BARE_JWT, id="bare_jwt"),
]


@pytest.mark.parametrize("secret_text", SECRET_SHAPES)
def test_gate_refuses_every_real_secret_shape(secret_text: str) -> None:
    assert secret_gate.contains_secret_shaped_content(secret_text) is True


def test_gate_still_admits_ordinary_output() -> None:
    assert (
        secret_gate.contains_secret_shaped_content(
            "nothing sensitive here, just plain command output"
        )
        is False
    )


def test_file_gate_refuses_a_pem_block_on_disk(tmp_path) -> None:
    path = tmp_path / "id_rsa"
    path.write_text(PEM_RSA)
    assert secret_gate.file_has_secret_shaped_content(path) is True
