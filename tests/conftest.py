"""Test-wide safety nets, applied to every test automatically.

1. No network: any attempt to resolve or connect to a non-local host raises RuntimeError,
   so a test that accidentally reaches yfinance fails loudly instead of passing by luck.
2. No real cache: OPTIONLAB_CACHE_DIR points at a temporary folder, so tests never read or
   write data/cache/. Tests that pass cache_dir=tmp_path explicitly are unaffected.
"""
import socket

import pytest

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", None, ""}


def _blocked(host) -> RuntimeError:
    return RuntimeError(f"Tests must not use the network (tried to reach {host!r}).")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else None   # str/bytes = local socket file
        if host not in _LOCAL_HOSTS:
            raise _blocked(host)
        return real_connect(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in _LOCAL_HOSTS:
            raise _blocked(host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(scope="session")
def _session_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("optionlab_cache")


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, _session_cache_dir):
    monkeypatch.setenv("OPTIONLAB_CACHE_DIR", str(_session_cache_dir))
