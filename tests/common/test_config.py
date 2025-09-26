import os
from dataclasses import dataclass, field
import typing

import pytest

from mrt.common.config import MRTConfig, LogConfig, _BaseConfig

@dataclass
class MyConfig(MRTConfig):
    my_str: str = "default"
    my_int: int = 10
    my_bool: bool = False
    my_list: typing.List[str] = field(default_factory=list)

def test_load_from_env():
    # Test case 1: No environment variables set
    cfg = MyConfig()
    new_cfg = cfg.copy().load_from_env()
    assert new_cfg == cfg # Should return self
    assert cfg.my_str == "default"
    assert cfg.my_int == 10
    assert cfg.my_bool is False
    assert cfg.my_list == []
    assert cfg.frontend == "pytorch" # from parent

    # Test case 2: Environment variables are set
    os.environ["MY_STR"] = "hello"
    os.environ["MY_INT"] = "42"
    os.environ["MY_BOOL"] = "true"
    os.environ["MY_LIST"] = "a,b,c"
    os.environ["FRONTEND"] = "tvm" # inherited attribute

    try:
        cfg = MyConfig()
        new_cfg = cfg.copy().load_from_env()

        assert new_cfg is not cfg # Should return a new instance
        assert new_cfg.my_str == "hello"
        assert new_cfg.my_int == 42
        assert new_cfg.my_bool is True
        assert new_cfg.my_list == ["a", "b", "c"]
        assert new_cfg.frontend == "tvm"
    finally:
        # Clean up environment variables
        del os.environ["MY_STR"]
        del os.environ["MY_INT"]
        del os.environ["MY_BOOL"]
        del os.environ["MY_LIST"]
        del os.environ["FRONTEND"]

def test_log_config_load_from_env():
    os.environ["LOG_VOT_CBS"] = "cb1,cb2"

    try:
        cfg = LogConfig()
        new_cfg = cfg.copy().load_from_env()

        assert new_cfg.log_vot_cbs == ["cb1", "cb2"]
    finally:
        del os.environ["LOG_VOT_CBS"]

def test_empty_env_does_not_mutate():
    cfg = MRTConfig()
    new_cfg = cfg.copy().load_from_env()
    assert cfg == new_cfg

def test_config_scope_enter_exit():
    orig = MRTConfig.G()
    orig_frontend = orig.frontend
    with MRTConfig(frontend="tvm") as cfg:
        assert MRTConfig.G() is cfg
        assert MRTConfig.G().frontend == "tvm"
    after = MRTConfig.G()
    assert after is orig
    assert after.frontend == orig_frontend

def test_nested_same_class_scope():
    base = MRTConfig.G()
    with MRTConfig(frontend="tvm") as c1:
        assert MRTConfig.G() is c1
        with MRTConfig(frontend="pytorch") as c2:
            assert MRTConfig.G() is c2
            assert MRTConfig.G().frontend == "pytorch"
        assert MRTConfig.G() is c1
        assert MRTConfig.G().frontend == "tvm"
    assert MRTConfig.G() is base

def test_scope_independent_between_classes():
    orig_mrt = MRTConfig.G()
    orig_log = LogConfig.G()
    with MRTConfig(frontend="tvm"):
        assert LogConfig.G() is orig_log
    with LogConfig(name_width=20) as L:
        assert MRTConfig.G() is orig_mrt
        assert LogConfig.G() is L
    assert MRTConfig.G() is orig_mrt
    assert LogConfig.G() is orig_log

if __name__ == "__main__":
    pytest.main(["-s", __file__, "-vvv"])
