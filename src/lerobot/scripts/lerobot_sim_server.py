"""Run a simulator server from an explicit YAML configuration."""

import argparse
import logging
import signal

from lerobot.sim_server.configuration import load_config
from lerobot.sim_server.server import ServerConfig, SimServer
from lerobot.sims.backend import BackendConfig
from lerobot.transport.zenoh import ZenohConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    data = load_config(args.config, args.set)
    data["sim"] = BackendConfig(**data["sim"])
    if "zenoh" in data:
        data["zenoh"] = ZenohConfig(**data["zenoh"])
    logging.basicConfig(level=logging.INFO)
    server = SimServer(ServerConfig(**data))
    signal.signal(signal.SIGTERM, lambda *_: server.stop.set())
    try:
        server.serve()
    except KeyboardInterrupt:
        server.stop.set()


if __name__ == "__main__":
    main()
