""".env loading and env-var override helpers in config.py."""

import os

from vcb.config import _env_bool, _load_dotenv


def test_env_bool_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("VCB_TEST_FLAG", raising=False)
    assert _env_bool("VCB_TEST_FLAG", True) is True
    assert _env_bool("VCB_TEST_FLAG", False) is False


def test_env_bool_parses_truthy_values(monkeypatch):
    for val in ("1", "true", "True", "yes", "on"):
        monkeypatch.setenv("VCB_TEST_FLAG", val)
        assert _env_bool("VCB_TEST_FLAG", False) is True
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("VCB_TEST_FLAG", val)
        assert _env_bool("VCB_TEST_FLAG", True) is False


def test_load_dotenv_missing_file_is_a_noop(tmp_path):
    _load_dotenv(tmp_path / "does_not_exist.env")   # must not raise


def test_load_dotenv_sets_unset_vars_only(tmp_path, monkeypatch):
    monkeypatch.delenv("VCB_TEST_A", raising=False)
    monkeypatch.setenv("VCB_TEST_B", "from_real_env")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        'VCB_TEST_A="from_dotenv"\n'
        "VCB_TEST_B=from_dotenv\n",
        encoding="utf-8",
    )
    _load_dotenv(env_file)
    assert os.environ["VCB_TEST_A"] == "from_dotenv"
    # A real environment variable always wins over .env.
    assert os.environ["VCB_TEST_B"] == "from_real_env"


def test_load_dotenv_strips_inline_comment(tmp_path, monkeypatch):
    monkeypatch.delenv("VCB_TEST_C", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "VCB_TEST_C=qwen3-4b                     # key into MODEL_PRESETS\n",
        encoding="utf-8",
    )
    _load_dotenv(env_file)
    assert os.environ["VCB_TEST_C"] == "qwen3-4b"


def test_load_dotenv_quoted_value_keeps_hash(tmp_path, monkeypatch):
    monkeypatch.delenv("VCB_TEST_D", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('VCB_TEST_D="not a # comment"\n', encoding="utf-8")
    _load_dotenv(env_file)
    assert os.environ["VCB_TEST_D"] == "not a # comment"


def test_load_dotenv_records_shadowed_values(tmp_path, monkeypatch):
    from vcb import config

    monkeypatch.setattr(config, "ENV_OVERRIDES", {})
    monkeypatch.setenv("VCB_TEST_E", "stale")
    monkeypatch.setenv("VCB_TEST_F", "same")
    env_file = tmp_path / ".env"
    env_file.write_text("VCB_TEST_E=fresh\nVCB_TEST_F=same\n", encoding="utf-8")
    config._load_dotenv(env_file)
    assert os.environ["VCB_TEST_E"] == "stale"          # real env still wins
    assert config.ENV_OVERRIDES == {"VCB_TEST_E": ("stale", "fresh")}
