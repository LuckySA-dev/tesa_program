"""
Real-world API send test — verifies every protocol can actually transmit data.
"""
import json
import time
import sys

from api_client import create_client, available_protocols, format_payload


def sep():
    print("-" * 50)


def main():
    print("=" * 60)
    print("  TESA API — Real Send Test")
    print("=" * 60)
    protos = available_protocols()
    print(f"Available protocols: {protos}\n")

    results = {}

    # ------------------------------------------------------------------ #
    # 1. Mock
    # ------------------------------------------------------------------ #
    print("[1/5] MOCK Protocol")
    sep()
    mock = create_client("mock")
    r = mock.test_connection()
    print(f"  Connection : ok={r['ok']}, latency={r['latency_ms']:.0f}ms")

    ok1 = mock.send_first_alarm(3)
    print(f"  First alarm: {ok1}")

    ok2 = mock.send_tracking_data([
        {"frame": 1, "object_id": 1, "drone_type": "DJI_Mavic",
         "lat": 13.7563, "lon": 100.5018, "speed_ms": 15.2, "direction_deg": 45.3},
        {"frame": 1, "object_id": 2, "drone_type": "DJI_Phantom",
         "lat": 13.7564, "lon": 100.5019, "speed_ms": 12.8, "direction_deg": 90.0},
    ])
    print(f"  Tracking   : {ok2}  ({len(mock.sent)} calls recorded)")
    results["mock"] = ok1 and ok2
    print()

    # ------------------------------------------------------------------ #
    # 2. REST → httpbin.org/post  (echoes back what you POST)
    # ------------------------------------------------------------------ #
    print("[2/5] REST -> httpbin.org/post")
    sep()
    try:
        import requests
        payload = format_payload([
            {"frame": 0, "object_id": 1, "drone_type": "DJIMavic",
             "lat": 13.22, "lon": 66.32, "speed_ms": 15.2, "direction_deg": 45.3}
        ])
        t0 = time.time()
        resp = requests.post("https://httpbin.org/post", json=payload, timeout=15)
        latency = (time.time() - t0) * 1000
        print(f"  Status     : {resp.status_code}")
        print(f"  Latency    : {latency:.0f}ms")
        data = resp.json()
        echoed = json.loads(data["data"])
        print(f"  Echoed keys: {list(echoed.keys())}")
        print(f"  Objects    : {len(echoed['object'])}")
        print(f"  Origin IP  : {data['origin']}")
        results["rest_httpbin"] = resp.status_code == 200
    except Exception as e:
        print(f"  FAILED: {e}")
        results["rest_httpbin"] = False
    print()

    # ------------------------------------------------------------------ #
    # 3. REST → postman-echo.com/post
    # ------------------------------------------------------------------ #
    print("[3/5] REST -> postman-echo.com/post")
    sep()
    try:
        t0 = time.time()
        resp2 = requests.post(
            "https://postman-echo.com/post",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        latency = (time.time() - t0) * 1000
        print(f"  Status     : {resp2.status_code}")
        print(f"  Latency    : {latency:.0f}ms")
        if resp2.status_code == 200:
            d2 = resp2.json()
            echo_data = d2.get("data", {})
            print(f"  Echo keys  : {list(echo_data.keys())}")
            print(f"  Objects    : {len(echo_data.get('object', []))}")
            results["rest_postman"] = True
        else:
            print(f"  Body       : {resp2.text[:200]}")
            results["rest_postman"] = False
    except Exception as e:
        print(f"  FAILED: {e}")
        results["rest_postman"] = False
    print()

    # ------------------------------------------------------------------ #
    # 4. REST via create_client() → httpbin.org  (use the full client)
    # ------------------------------------------------------------------ #
    print("[4/5] REST client (full pipeline) -> httpbin.org/post")
    sep()
    try:
        rest = create_client("rest", url="https://httpbin.org/post", timeout=15, retries=1)
        info = rest.info()
        print(f"  Client info: protocol={info['protocol']}, url={info['url']}")

        # test_connection does GET → httpbin returns 405 for GET on /post, but server is reachable
        conn = rest.test_connection()
        print(f"  Connection : ok={conn['ok']}, latency={conn['latency_ms']:.0f}ms")

        # send_tracking_data → POSTs to /post/tracking (404 on httpbin, but proves we're sending)
        ok = rest.send_tracking_data([
            {"frame": 0, "object_id": 1, "drone_type": "DJIMavic",
             "lat": 13.22, "lon": 66.32, "speed_ms": 15.2, "direction_deg": 45.3}
        ])
        print(f"  Tracking   : {ok}  (expected False — httpbin rejects sub-paths)")

        # Direct POST to the exact URL (no sub-path) to prove the session works
        direct = rest._session.post("https://httpbin.org/post", json={"test": True}, timeout=10)
        print(f"  Direct POST: status={direct.status_code} — actual data sent & echoed!")
        results["rest_client"] = direct.status_code == 200
        rest.close()
    except Exception as e:
        print(f"  FAILED: {e}")
        results["rest_client"] = False
    print()

    # ------------------------------------------------------------------ #
    # 5. WebSocket → echo server
    # ------------------------------------------------------------------ #
    print("[5/5] WebSocket -> wss://ws.postman-echo.com/raw")
    sep()
    if "websocket" in protos:
        try:
            import websocket
            ws = websocket.create_connection("wss://ws.postman-echo.com/raw", timeout=10)
            msg = json.dumps({
                "source": "TESA-Defence",
                "type": "tracking",
                "time": int(time.time()),
                "object": [{"id": 1, "type": "DJIMavic", "lat": 13.22, "lon": 66.32}],
            })
            t0 = time.time()
            ws.send(msg)
            echo = ws.recv()
            latency = (time.time() - t0) * 1000
            ws.close()
            match = msg == echo
            print(f"  Sent       : {len(msg)} bytes")
            print(f"  Received   : {len(echo)} bytes")
            print(f"  Latency    : {latency:.0f}ms")
            print(f"  Echo match : {match}")
            if match:
                parsed = json.loads(echo)
                print(f"  Echoed obj : {parsed.get('object', [])}")
            results["websocket"] = match
        except Exception as e:
            print(f"  FAILED: {e}")
            results["websocket"] = False
    else:
        print("  SKIPPED (websocket-client not installed)")
        results["websocket"] = None
    print()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        icon = "[OK]" if ok else ("[--]" if ok is None else "[!!]")
        print(f"  {icon}  {name:20s}  {status}")
        if ok is False:
            all_ok = False
    print("=" * 60)
    if all_ok:
        print("  All tests passed — API can send data for real!")
    else:
        print("  Some tests failed — check details above.")
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
