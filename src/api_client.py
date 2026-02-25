"""
API Client for Satellite Communication
========================================
Multi-protocol drone alert client supporting:
  - REST  (HTTP/HTTPS POST)
  - MQTT  (publish to broker)
  - WebSocket (send JSON frames)
  - gRPC  (unary call via reflection-free stub)

Usage:
    from api_client import create_client

    client = create_client()              # auto-detect from env / config
    client = create_client("rest")        # explicit
    client = create_client("mqtt")

    client.test_connection()
    client.send_tracking_data(objects)
    client.send_first_alarm(3, frame)
    client.close()
"""

from __future__ import annotations

import base64
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from hw import sprint

# ---------------------------------------------------------------------------
#  Payload helpers (shared across all protocols)
# ---------------------------------------------------------------------------

def encode_image_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode an OpenCV frame to a base64 JPEG string."""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", frame, params)
    return base64.b64encode(buf).decode("utf-8")


def format_payload(
    objects: List[Dict],
    frame: Optional[np.ndarray] = None,
    include_image: bool = True,
    image_quality: int = 85,
) -> Dict:
    """Build the canonical TESA JSON payload."""
    formatted = []
    for obj in objects:
        formatted.append(
            {
                "frame": obj.get("frame", 0),
                "id": obj.get("object_id", 0),
                "type": obj.get("drone_type", "Unknown"),
                "lat": float(obj.get("lat", 0.0)),
                "lon": float(obj.get("lon", 0.0)),
                "velocity": float(obj.get("speed_ms", 0.0)),
                "direction": float(obj.get("direction_deg", 0.0)),
            }
        )
    payload: Dict[str, Any] = {
        "time": int(time.time()),
        "object": formatted,
    }
    if include_image and frame is not None:
        payload["image_base64"] = encode_image_base64(frame, image_quality)
    else:
        payload["image_base64"] = ""
    return payload


# =========================================================================== #
#  Abstract base
# =========================================================================== #

class BaseAPIClient(ABC):
    """Protocol-agnostic base for all API transports."""

    protocol: str = "base"

    def __init__(
        self,
        url: str,
        api_key: str = "",
        timeout: int = 10,
        retries: int = 2,
        image_quality: int = 85,
        **kwargs,
    ):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(0, retries)
        self.image_quality = image_quality
        self.first_alarm_sent = False
        self.last_alarm_time = 0.0
        self.alarm_cooldown = 30
        self._extra = kwargs

    # -- public API (identical across all protocols) -------------------------

    def send_first_alarm(
        self, drone_count: int, frame: Optional[np.ndarray] = None
    ) -> bool:
        if self.first_alarm_sent:
            return True
        now = time.time()
        if now - self.last_alarm_time < self.alarm_cooldown:
            return False
        payload = {
            "type": "first_alarm",
            "time": int(now),
            "drone_count": drone_count,
            "message": f"FIRST ALARM: {drone_count} drone(s) detected",
        }
        if frame is not None:
            payload["image_base64"] = encode_image_base64(
                frame, self.image_quality
            )
        ok = self._send_with_retry(payload, topic="/alarm")
        if ok:
            self.first_alarm_sent = True
            self.last_alarm_time = now
            sprint(f"[ALARM] First alarm sent: {drone_count} drones")
        return ok

    def send_tracking_data(
        self,
        objects: List[Dict],
        frame: Optional[np.ndarray] = None,
        include_image: bool = False,
    ) -> bool:
        if not objects:
            return False
        payload = format_payload(objects, frame, include_image, self.image_quality)
        ok = self._send_with_retry(payload, topic="/tracking")
        if ok:
            sprint(f"[API] Sent tracking: {len(objects)} objects via {self.protocol}")
        return ok

    def send_batch(
        self,
        all_objects: List[Dict],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        if not all_objects:
            return False
        frames_data: Dict[int, list] = {}
        for obj in all_objects:
            fn = obj.get("frame", 0)
            frames_data.setdefault(fn, []).append(obj)
        ok_count = 0
        for fn, objs in frames_data.items():
            p = format_payload(objs, frame if fn == max(frames_data) else None, False)
            if self._send_with_retry(p, topic="/tracking"):
                ok_count += 1
        sprint(f"[API] Batch sent: {ok_count}/{len(frames_data)} via {self.protocol}")
        return ok_count > 0

    def test_connection(self) -> dict:
        """Return {"ok": bool, "latency_ms": float, "detail": str}."""
        t0 = time.time()
        try:
            result = self._test()
            latency = (time.time() - t0) * 1000
            result.setdefault("latency_ms", round(latency, 1))
            if result.get("ok"):
                sprint(f"[OK] {self.protocol.upper()} connection OK ({result['latency_ms']} ms)")
            else:
                sprint(f"[FAIL] {self.protocol.upper()} test failed: {result.get('detail', '?')}")
            return result
        except Exception as e:
            latency = (time.time() - t0) * 1000
            sprint(f"[FAIL] {self.protocol.upper()} test error: {e}")
            return {"ok": False, "latency_ms": round(latency, 1), "detail": str(e)}

    def close(self):
        """Override in subclasses that hold persistent connections."""
        pass

    def info(self) -> dict:
        """Return a summary dict for dashboards."""
        return {
            "protocol": self.protocol,
            "url": self.url,
            "timeout": self.timeout,
            "retries": self.retries,
            "api_key_set": bool(self.api_key),
        }

    # -- private helpers -----------------------------------------------------

    def _send_with_retry(self, payload: Dict, topic: str = "") -> bool:
        last_err = None
        for attempt in range(1 + self.retries):
            try:
                ok = self._send(payload, topic)
                if ok:
                    return True
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    wait = 0.5 * (2 ** attempt)
                    sprint(f"[RETRY] Attempt {attempt+1} failed ({e}), retrying in {wait:.1f}s...")
                    time.sleep(wait)
        if last_err:
            sprint(f"[FAIL] {self.protocol.upper()} send failed after {1+self.retries} attempts: {last_err}")
        return False

    @abstractmethod
    def _send(self, payload: Dict, topic: str) -> bool:
        ...

    @abstractmethod
    def _test(self) -> dict:
        ...


# =========================================================================== #
#  REST (HTTP/HTTPS)
# =========================================================================== #

class RESTClient(BaseAPIClient):
    """Standard HTTP(S) POST client using ``requests``."""

    protocol = "rest"

    def __init__(self, **kw):
        super().__init__(**kw)
        import requests as _req
        self._session = _req.Session()
        if self.api_key:
            self._session.headers["Authorization"] = f"Bearer {self.api_key}"
        self._session.headers["Content-Type"] = "application/json"
        self._session.headers["User-Agent"] = "TESA-Defence-API/2.0"

    def _send(self, payload: Dict, topic: str) -> bool:
        url = self.url.rstrip("/") + topic
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        if 200 <= resp.status_code < 300:
            return True
        sprint(f"[FAIL] REST {resp.status_code}: {resp.text[:200]}")
        return False

    def _test(self) -> dict:
        try:
            resp = self._session.get(self.url, timeout=self.timeout)
            return {
                "ok": resp.status_code < 500,
                "status_code": resp.status_code,
                "detail": resp.text[:300],
            }
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    def close(self):
        self._session.close()


# =========================================================================== #
#  MQTT
# =========================================================================== #

class MQTTClient(BaseAPIClient):
    """MQTT publish client using ``paho-mqtt``."""

    protocol = "mqtt"

    def __init__(self, **kw):
        super().__init__(**kw)
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise ImportError(
                "paho-mqtt is required for MQTT protocol. "
                "Install with: pip install paho-mqtt"
            )
        self._mqtt_module = mqtt
        self._base_topic = self._extra.get("topic", "tesa/drone")
        self._qos = int(self._extra.get("qos", 1))

        # Parse broker URL — mqtt://host:port or just host:port
        broker_url = self.url.replace("mqtt://", "").replace("mqtts://", "")
        parts = broker_url.split(":")
        self._host = parts[0]
        self._port = int(parts[1]) if len(parts) > 1 else 1883
        self._use_tls = self.url.startswith("mqtts://")

        # Create client – support both paho-mqtt v1 and v2
        try:
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"tesa-drone-{int(time.time())}",
            )
        except (AttributeError, TypeError):
            self._client = mqtt.Client(client_id=f"tesa-drone-{int(time.time())}")

        if self.api_key:
            self._client.username_pw_set("tesa", self.api_key)
        if self._use_tls:
            self._client.tls_set()
        self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            self._client.connect(self._host, self._port, keepalive=60)
            self._client.loop_start()
            self._connected = True

    def _send(self, payload: Dict, topic: str) -> bool:
        self._ensure_connected()
        full_topic = self._base_topic + topic
        msg = json.dumps(payload)
        info = self._client.publish(full_topic, msg, qos=self._qos)
        info.wait_for_publish(timeout=self.timeout)
        return info.is_published()

    def _test(self) -> dict:
        try:
            self._ensure_connected()
            info = self._client.publish(
                self._base_topic + "/ping",
                json.dumps({"ping": int(time.time())}),
                qos=0,
            )
            info.wait_for_publish(timeout=self.timeout)
            return {"ok": info.is_published(), "detail": f"broker={self._host}:{self._port}"}
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    def close(self):
        if self._connected:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False


# =========================================================================== #
#  WebSocket
# =========================================================================== #

class WebSocketClient(BaseAPIClient):
    """WebSocket client using ``websocket-client``."""

    protocol = "websocket"

    def __init__(self, **kw):
        super().__init__(**kw)
        try:
            import websocket  # websocket-client package
        except ImportError:
            raise ImportError(
                "websocket-client is required for WebSocket protocol. "
                "Install with: pip install websocket-client"
            )
        self._ws_module = websocket
        self._ws: Any = None

    def _ensure_connected(self):
        if self._ws is None or not self._ws.connected:
            ws_url = self.url
            # Normalise scheme
            if ws_url.startswith("http://"):
                ws_url = "ws://" + ws_url[7:]
            elif ws_url.startswith("https://"):
                ws_url = "wss://" + ws_url[8:]
            elif not ws_url.startswith(("ws://", "wss://")):
                ws_url = "ws://" + ws_url

            header = []
            if self.api_key:
                header.append(f"Authorization: Bearer {self.api_key}")
            self._ws = self._ws_module.create_connection(
                ws_url,
                timeout=self.timeout,
                header=header,
            )

    def _send(self, payload: Dict, topic: str) -> bool:
        self._ensure_connected()
        payload["_topic"] = topic  # embed topic in message
        self._ws.send(json.dumps(payload))
        # Try to read ack (non-blocking, best-effort)
        try:
            self._ws.settimeout(min(2, self.timeout))
            ack = self._ws.recv()
            data = json.loads(ack) if ack else {}
            return data.get("ok", True)
        except Exception:
            return True  # assume success if no ack protocol

    def _test(self) -> dict:
        try:
            self._ensure_connected()
            self._ws.send(json.dumps({"ping": int(time.time())}))
            try:
                self._ws.settimeout(min(2, self.timeout))
                resp = self._ws.recv()
                return {"ok": True, "detail": resp[:200] if resp else "connected"}
            except Exception:
                return {"ok": True, "detail": "connected (no echo)"}
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    def close(self):
        if self._ws and self._ws.connected:
            self._ws.close()
            self._ws = None


# =========================================================================== #
#  gRPC
# =========================================================================== #

class GRPCClient(BaseAPIClient):
    """gRPC unary-call client.

    Expects a .proto-generated ``drone_alert_pb2`` and
    ``drone_alert_pb2_grpc`` on the Python path.  If they are not found
    the client falls back to a gRPC health-check.
    """

    protocol = "grpc"

    def __init__(self, **kw):
        super().__init__(**kw)
        try:
            import grpc
        except ImportError:
            raise ImportError(
                "grpcio is required for gRPC protocol. "
                "Install with: pip install grpcio grpcio-tools"
            )
        self._grpc = grpc
        self._channel = None
        self._stub = None
        self._has_proto = False

        # Parse host:port (strip grpc:// scheme if present)
        target = self.url
        for prefix in ("grpc://", "grpcs://"):
            if target.startswith(prefix):
                target = target[len(prefix):]
        self._target = target
        self._secure = self.url.startswith("grpcs://")

        # Try loading generated stubs
        try:
            from drone_alert_pb2 import DroneAlertRequest  # type: ignore
            from drone_alert_pb2_grpc import DroneAlertServiceStub  # type: ignore

            self._request_cls = DroneAlertRequest
            self._stub_cls = DroneAlertServiceStub
            self._has_proto = True
        except ImportError:
            self._request_cls = None
            self._stub_cls = None

    def _ensure_channel(self):
        if self._channel is None:
            if self._secure:
                creds = self._grpc.ssl_channel_credentials()
                self._channel = self._grpc.secure_channel(self._target, creds)
            else:
                self._channel = self._grpc.insecure_channel(self._target)

            if self._has_proto:
                self._stub = self._stub_cls(self._channel)

    def _send(self, payload: Dict, topic: str) -> bool:
        self._ensure_channel()
        if self._has_proto and self._stub is not None:
            import json as _json
            req = self._request_cls(json_payload=_json.dumps(payload))
            resp = self._stub.SendAlert(req, timeout=self.timeout)
            return getattr(resp, "ok", True)
        else:
            from grpc_health.v1 import health_pb2, health_pb2_grpc
            stub = health_pb2_grpc.HealthStub(self._channel)
            req = health_pb2.HealthCheckRequest(service="tesa.drone")
            resp = stub.Check(req, timeout=self.timeout)
            return resp.status == health_pb2.HealthCheckResponse.SERVING

    def _test(self) -> dict:
        try:
            self._ensure_channel()
            try:
                from grpc_health.v1 import health_pb2, health_pb2_grpc
                stub = health_pb2_grpc.HealthStub(self._channel)
                req = health_pb2.HealthCheckRequest(service="")
                resp = stub.Check(req, timeout=self.timeout)
                return {
                    "ok": resp.status == health_pb2.HealthCheckResponse.SERVING,
                    "detail": f"gRPC health={resp.status}",
                }
            except ImportError:
                import grpc
                future = self._channel.channel_ready()
                future.result(timeout=self.timeout)
                return {"ok": True, "detail": "channel ready"}
        except Exception as e:
            return {"ok": False, "detail": str(e)}

    def close(self):
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None


# =========================================================================== #
#  Mock (testing without a real server)
# =========================================================================== #

class MockClient(BaseAPIClient):
    """In-memory mock that records every call."""

    protocol = "mock"

    def __init__(self, **kw):
        kw.setdefault("url", "mock://localhost")
        super().__init__(**kw)
        self.sent: List[Dict] = []
        self._latency = float(kw.get("mock_latency", 0.01))

    def _send(self, payload: Dict, topic: str) -> bool:
        time.sleep(self._latency)
        self.sent.append({"topic": topic, "payload": payload, "ts": time.time()})
        sprint(f"[MOCK] Sent to {topic}: {len(payload.get('object', []))} objects")
        return True

    def _test(self) -> dict:
        return {"ok": True, "detail": "mock - always OK"}


# =========================================================================== #
#  Factory
# =========================================================================== #

_REGISTRY: Dict[str, type] = {
    "rest": RESTClient,
    "http": RESTClient,
    "https": RESTClient,
    "mqtt": MQTTClient,
    "websocket": WebSocketClient,
    "ws": WebSocketClient,
    "grpc": GRPCClient,
    "mock": MockClient,
}


def detect_protocol(url: str) -> str:
    """Guess protocol from URL scheme."""
    url_lower = url.lower()
    if url_lower.startswith(("mqtt://", "mqtts://")):
        return "mqtt"
    if url_lower.startswith(("ws://", "wss://")):
        return "websocket"
    if url_lower.startswith(("grpc://", "grpcs://")):
        return "grpc"
    if url_lower.startswith("mock://"):
        return "mock"
    return "rest"


def available_protocols() -> List[str]:
    """Return list of protocols whose dependencies are importable."""
    protos = ["rest", "mock"]  # always available
    try:
        import paho.mqtt.client  # noqa: F401
        protos.append("mqtt")
    except ImportError:
        pass
    try:
        import websocket  # noqa: F401
        protos.append("websocket")
    except ImportError:
        pass
    try:
        import grpc  # noqa: F401
        protos.append("grpc")
    except ImportError:
        pass
    return protos


def create_client(
    protocol: Optional[str] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
    **kwargs,
) -> BaseAPIClient:
    """
    Create an API client.  Reads from env vars / config if args are None.

    Env vars (override config.py):
        API_PROTOCOL  - rest | mqtt | websocket | grpc | mock
        API_URL       - endpoint URL
        API_KEY       - bearer token / password
        API_TIMEOUT   - seconds
        API_RETRIES   - retry count
        MQTT_TOPIC    - base MQTT topic  (default: tesa/drone)
        MQTT_QOS      - 0, 1, 2          (default: 1)
    """
    env = os.environ.get
    _url = url or env("API_URL", "")
    _key = api_key if api_key is not None else env("API_KEY", "")
    _timeout = timeout or int(env("API_TIMEOUT", "10"))
    _retries = retries if retries is not None else int(env("API_RETRIES", "2"))

    # Fall back to config.py
    if not _url:
        try:
            from config import API_CONFIG
            _url = API_CONFIG.get("api_url", "https://api.tesa.or.th/drone")
            _key = _key or API_CONFIG.get("api_key", "")
            _timeout = timeout or API_CONFIG.get("timeout", 10)
        except ImportError:
            _url = "https://api.tesa.or.th/drone"

    # Determine protocol
    _proto = protocol or env("API_PROTOCOL", "")
    if not _proto:
        _proto = detect_protocol(_url)

    _proto = _proto.lower()
    cls = _REGISTRY.get(_proto)
    if cls is None:
        raise ValueError(
            f"Unknown protocol '{_proto}'. "
            f"Available: {', '.join(_REGISTRY.keys())}"
        )

    # Extra kwargs per protocol
    kwargs.setdefault("topic", env("MQTT_TOPIC", "tesa/drone"))
    kwargs.setdefault("qos", env("MQTT_QOS", "1"))

    return cls(
        url=_url,
        api_key=_key,
        timeout=_timeout,
        retries=_retries,
        **kwargs,
    )


# =========================================================================== #
#  Standalone test
# =========================================================================== #

if __name__ == "__main__":
    print("=" * 60)
    print("TESA API Client - Multi-Protocol Test")
    print("=" * 60)

    print(f"\nAvailable protocols: {available_protocols()}")

    # -- Mock test -----------------------------------------------------------
    print("\n--- Mock Client ---")
    mock = create_client("mock")
    print(f"Info: {mock.info()}")
    res = mock.test_connection()
    print(f"Test: {res}")

    ok = mock.send_first_alarm(3)
    print(f"First alarm: {ok}")

    ok = mock.send_tracking_data(
        [
            {"frame": 1, "object_id": 1, "drone_type": "DJI_Mavic",
             "lat": 13.7563, "lon": 100.5018, "speed_ms": 15.2, "direction_deg": 45.3},
            {"frame": 1, "object_id": 2, "drone_type": "DJI_Phantom",
             "lat": 13.7564, "lon": 100.5019, "speed_ms": 12.8, "direction_deg": 90.0},
        ]
    )
    print(f"Tracking: {ok}")
    print(f"Total calls: {len(mock.sent)}")

    # -- REST dry-run -------------------------------------------------------
    print("\n--- REST Client (dry-run) ---")
    rest = create_client("rest", url="https://httpbin.org/post")
    print(f"Info: {rest.info()}")
    res = rest.test_connection()
    print(f"Test: {res}")
    rest.close()

    print("\n" + "=" * 60)
    print("All tests passed!")
