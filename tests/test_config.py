# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment timestamps must not change the process-wide timezone."""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from utils import config


@pytest.fixture
def work_dir(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "model_config.yaml").write_text(
        "defaults:\n  model_name: text-default\n  image_model_name: image-default\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture(autouse=True)
def restore_timezone():
    original = os.environ.get("TZ")
    yield
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    if hasattr(time, "tzset"):
        time.tzset()


@pytest.mark.parametrize("host_tz", [None, "UTC0", "JST-9"])
def test_configuration_preserves_host_timezone(work_dir, host_tz, monkeypatch):
    if host_tz is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", host_tz)
    if hasattr(time, "tzset"):
        time.tzset()
    before = (os.environ.get("TZ"), time.tzname, time.timezone, time.localtime(0))
    experiment = config.ExpConfig("PaperBananaBench", work_dir=work_dir)
    assert (
        os.environ.get("TZ"),
        time.tzname,
        time.timezone,
        time.localtime(0),
    ) == before
    assert experiment.result_dir.is_dir()


def test_creation_without_tzset(work_dir, monkeypatch):
    monkeypatch.delattr(time, "tzset", raising=False)
    experiment = config.ExpConfig("PaperBananaBench", work_dir=work_dir)
    assert len(experiment.timestamp) == 9
    assert experiment.model_name == "text-default"


@pytest.mark.parametrize(
    "instant,expected",
    [
        ("2026-01-01T07:30:00+00:00", "1231_2330"),
        ("2026-07-01T07:30:00+00:00", "0701_0030"),
        ("2026-03-08T09:59:00+00:00", "0308_0159"),
        ("2026-03-08T10:00:00+00:00", "0308_0300"),
        ("2026-11-01T08:30:00+00:00", "1101_0130"),
        ("2026-11-01T09:30:00+00:00", "1101_0130"),
    ],
)
def test_timestamp_uses_los_angeles_rules(work_dir, instant, expected, monkeypatch):
    fixed = datetime.fromisoformat(instant)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz)

    # Freeze only the wall clock; zone conversion and formatting remain real.
    monkeypatch.setattr(config, "datetime", FrozenDatetime, raising=False)
    legacy_clock = mock.Mock(wraps=time)
    legacy_clock.strftime.side_effect = lambda fmt: fixed.astimezone(
        ZoneInfo("America/Los_Angeles")
    ).strftime(fmt)
    monkeypatch.setattr(config, "time", legacy_clock, raising=False)
    experiment = config.ExpConfig("PaperBananaBench", work_dir=work_dir)
    assert experiment.timestamp == expected
    assert experiment.exp_name == f"{expected}_autoret__test"


@pytest.mark.parametrize("timestamp", ["replay-01", ""])
def test_explicit_timestamp_is_preserved_without_clock_access(work_dir, timestamp):
    with mock.patch.object(config, "datetime", create=True) as clock:
        clock.now.side_effect = AssertionError("explicit timestamps need no clock")
        experiment = config.ExpConfig(
            "PaperBananaBench", timestamp=timestamp, work_dir=work_dir
        )
    assert experiment.timestamp == timestamp
    clock.now.assert_not_called()


@pytest.mark.parametrize(
    "text_model,image_model", [("", ""), ("explicit", ""), ("", "explicit")]
)
def test_yaml_model_defaults_and_output_path(work_dir, text_model, image_model):
    experiment = config.ExpConfig(
        "PaperBananaBench",
        task_name="plot",
        split_name="validation",
        retrieval_setting="manual",
        exp_mode="dev_full",
        timestamp="run",
        model_name=text_model,
        image_model_name=image_model,
        work_dir=work_dir,
    )
    assert experiment.model_name == (text_model or "text-default")
    assert experiment.image_model_name == (image_model or "image-default")
    assert experiment.exp_name == "run_manualret_dev_full_validation"
    assert experiment.result_dir == work_dir / "results" / "PaperBananaBench_plot"
    assert experiment.result_dir.is_dir()


def test_automatic_timestamp_uses_real_clock(work_dir):
    before = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles"))
    experiment = config.ExpConfig("PaperBananaBench", work_dir=work_dir)
    after = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles"))
    assert experiment.timestamp in {
        before.strftime("%m%d_%H%M"),
        after.strftime("%m%d_%H%M"),
    }


def test_tzdata_fallback_without_system_database(work_dir):
    code = """
from zoneinfo import ZoneInfo, reset_tzpath
from pathlib import Path
from utils.config import ExpConfig
reset_tzpath(())
ZoneInfo.clear_cache()
experiment = ExpConfig('PaperBananaBench', work_dir=Path(__import__('sys').argv[1]))
assert len(experiment.timestamp) == 9
"""
    subprocess.run([sys.executable, "-c", code, str(work_dir)], check=True)


@pytest.mark.parametrize("empty_file", [False, True])
def test_missing_or_empty_yaml_retains_empty_model_defaults(work_dir, empty_file):
    config_file = work_dir / "configs" / "model_config.yaml"
    if empty_file:
        config_file.write_text("", encoding="utf-8")
    else:
        config_file.unlink()
    experiment = config.ExpConfig(
        "PaperBananaBench", work_dir=work_dir, timestamp="run"
    )
    assert experiment.model_name == experiment.image_model_name == ""


def test_explicit_models_do_not_read_yaml(work_dir):
    (work_dir / "configs" / "model_config.yaml").write_text(
        "[malformed", encoding="utf-8"
    )
    experiment = config.ExpConfig(
        "PaperBananaBench",
        work_dir=work_dir,
        timestamp="run",
        model_name="chosen-text",
        image_model_name="chosen-image",
    )
    assert experiment.model_name == "chosen-text"
    assert experiment.image_model_name == "chosen-image"
