"""The conftest.py guard really blocks the network, so 'tests need no network' is enforced."""
import socket

import pytest


def test_connecting_to_a_remote_host_is_blocked():
    with pytest.raises(RuntimeError, match="Tests must not use the network"):
        socket.create_connection(("example.com", 80), timeout=1)


def test_resolving_a_remote_host_is_blocked():
    with pytest.raises(RuntimeError, match="example.com"):
        socket.getaddrinfo("example.com", 443)


def test_localhost_is_still_allowed():
    assert socket.getaddrinfo("localhost", 80)       # local lookups do not touch the network
