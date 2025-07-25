from __future__ import annotations

import subprocess
import time
import typing

from contextlib import contextmanager

from loguru import logger

from mwe_db_access.config import settings


@contextmanager
def sshtunnel(name: str) -> typing.Generator:
    """
    Establish an SSH tunnel.

    This context manager sets up an SSH tunnel using configurations specified by a name
    from the config settings. It creates a tunnel that forwards a local port to a remote
    port on a remote host.

    Parameters
    ----------
    name : str
        The key to retrieve the SSH configuration from the settings.

    Yields
    ------
    subprocess.Popen
        The subprocess handling the SSH connection.

    Raises
    ------
    ValueError
        If the SSH configuration for the specified name does not exist.
    subprocess.SubprocessError
        If the process to establish the SSH tunnel cannot be started.

    Examples
    --------
    >>> with sshtunnel('example_tunnel') as tunnel_proc:
    ...     # Operations using the tunnel can be performed here
    ...     pass
    """
    # Retrieve SSH configuration using the provided name
    ssh_config = settings.get("ssh-tunnel", {}).get(name)

    if ssh_config is None:
        msg = f"Missing ssh config for the specified {name=}."
        logger.error(msg)
        raise ValueError(msg)

    try:
        # Log the initiation of the SSH tunnel
        logger.info("Opening SSH tunnel...")
        proc = subprocess.Popen(
            [
                "ssh",
                "-N",
                "-L",
                f"{ssh_config['LOCAL_PORT']}:{ssh_config['LOCAL_ADDRESS']}:"
                f"{ssh_config['REMOTE_PORT']}",
                f"{ssh_config['USER']}@{ssh_config['HOST']}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Allow some time for the tunnel to establish
        time.sleep(3)

        # Check if process has terminated unexpectedly
        if proc.poll() is not None:
            raise subprocess.SubprocessError("Failed to establish the SSH tunnel.")

        yield proc

    finally:
        # Log the closure of the SSH tunnel
        logger.info("Closing SSH tunnel...")
        proc.kill()

        # Wait before fetching process output to ensure all resources are released
        time.sleep(3)
        outs, errs = proc.communicate()

        # Log the outputs from the SSH process
        logger.info(
            f"SSH process output: {outs.decode('utf-8')}, errors: "
            f"{errs.decode('utf-8')}"
        )
