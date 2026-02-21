"""
mDNS service announcer for Imagine Server.

Registers _imagine._tcp.local. service via zeroconf so that
Electron clients on the same LAN can auto-discover the server.

zeroconf is an optional dependency — if not installed, mDNS is silently disabled.
"""

import logging
import socket

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    """Get primary LAN IPv4 address via UDP socket trick (no actual traffic)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class ImagineServiceAnnouncer:
    """Register/unregister _imagine._tcp.local. mDNS service."""

    SERVICE_TYPE = "_imagine._tcp.local."

    def __init__(self, port: int, server_name: str | None = None):
        self.port = port
        self.server_name = server_name or socket.gethostname()
        self._zeroconf = None
        self._info = None

    def start(self):
        """Register mDNS service."""
        from zeroconf import Zeroconf, ServiceInfo

        local_ip = _get_local_ip()
        hostname = socket.gethostname()

        self._info = ServiceInfo(
            type_=self.SERVICE_TYPE,
            name=f"Imagine-{self.server_name}.{self.SERVICE_TYPE}",
            port=self.port,
            properties={
                "version": "4.0.0",
                "name": self.server_name,
            },
            parsed_addresses=[local_ip],
            server=f"{hostname}.local.",
        )

        self._zeroconf = Zeroconf()
        self._zeroconf.register_service(self._info)
        logger.info(
            f"mDNS registered: {self.server_name} at {local_ip}:{self.port}"
        )

    def stop(self):
        """Unregister mDNS service."""
        if self._zeroconf and self._info:
            try:
                self._zeroconf.unregister_service(self._info)
                self._zeroconf.close()
            except Exception as e:
                logger.warning(f"mDNS unregister error: {e}")
            self._zeroconf = None
            self._info = None
            logger.info("mDNS service unregistered")
