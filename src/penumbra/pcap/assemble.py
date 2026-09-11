"""pcap → UNSW-NB15 flow features.

This removes the objection that kills most ML-IDS projects in the Q&A: *"it only works on a CSV
somebody else prepared."* Point it at a capture and it produces the same feature vector the model
was trained on, so the detector runs on traffic rather than on a benchmark.

## Which features, and the ones we refuse to fake

UNSW-NB15's 42 features split cleanly along a line this project already committed to. Most are
functions of packet headers and timing — we compute those. A handful require reading the
application payload, and `docs/ETHICS_SCOPE.md` and the model card both state that Penumbra
inspects flow metadata only, never packet contents.

So those are emitted as zero and **named**, rather than silently filled:

| feature | why it is zero here |
|---|---|
| `service` | requires DPI to identify the protocol beyond a port guess |
| `trans_depth` | HTTP pipeline depth — needs the HTTP body |
| `response_body_len` | needs the HTTP body |
| `is_ftp_login` | needs the FTP control channel contents |
| `ct_ftp_cmd` | needs the FTP command stream |
| `ct_flw_http_mthd` | needs the HTTP method line |

That is a real limitation and it belongs on the slide, not in a footnote: a flow assembled from a
pcap is not identical to a UNSW-NB15 row, and the model will score it slightly differently. The
alternative — guessing `service` from the destination port and calling it the same feature — would
produce a number that looks comparable and is not.

`sloss`/`dloss` are also zero: retransmission counting needs full TCP sequence-space reassembly,
which is a correctness problem of its own and not one to solve approximately.

## Bidirectional flows

A flow is keyed on the *normalised* 5-tuple, so a packet and its reply land in the same flow, with
the endpoint that sent the first packet designated the source. Getting this wrong splits every
conversation into two half-flows, each with zero bytes in one direction — which looks remarkably
like a scan, to a model trained on real conversations.

Flows close on a 120-second idle timeout, on TCP FIN/RST, or at a 600-second active timeout. Those
are the conventional NetFlow values; a flow that never closes is a flow that never gets scored.

## Scope

Read-only. This package parses capture files and never transmits a packet. Captures must come from
a network you own — see `docs/ETHICS_SCOPE.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Emitted as zero because computing them honestly needs payload inspection, which this project
# does not do. Callers get this list so the gap is reportable rather than invisible.
PAYLOAD_DEPENDENT: tuple[str, ...] = (
    "service",
    "trans_depth",
    "response_body_len",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
)

# Needs full TCP sequence-space reassembly to count retransmissions. Approximating it would put a
# plausible number in a column the model weights, which is worse than a zero it can learn around.
UNAVAILABLE: tuple[str, ...] = ("sloss", "dloss")

IDLE_TIMEOUT = 120.0
ACTIVE_TIMEOUT = 600.0

# UNSW's ct_* columns are counts over a trailing window of this many connections, per the original
# Argus/Bro pipeline. Reproducing the window size matters: a count over 1,000 connections is not
# the same feature under a different name.
CT_WINDOW = 100


@dataclass(frozen=True)
class FlowKey:
    src: str
    sport: int
    dst: str
    dport: int
    proto: str

    @property
    def reverse(self) -> FlowKey:
        return FlowKey(self.dst, self.dport, self.src, self.sport, self.proto)


@dataclass
class Direction:
    """Counters for one half of a conversation."""

    packets: int = 0
    bytes: int = 0
    ttl: int = 0
    window: int = 0
    tcp_base_seq: int = 0
    first_time: float = 0.0
    last_time: float = 0.0
    gaps: list[float] = field(default_factory=list)

    def observe(self, when: float, size: int, ttl: int, window: int, seq: int) -> None:
        if self.packets == 0:
            self.first_time = when
            self.ttl = ttl
            self.window = window
            self.tcp_base_seq = seq
        else:
            self.gaps.append(when - self.last_time)
        self.packets += 1
        self.bytes += size
        self.last_time = when

    @property
    def mean_size(self) -> float:
        return self.bytes / self.packets if self.packets else 0.0

    @property
    def mean_gap_ms(self) -> float:
        return float(np.mean(self.gaps) * 1000.0) if self.gaps else 0.0

    @property
    def jitter_ms(self) -> float:
        """UNSW's `sjit`/`djit`: the variability of inter-packet arrival, in milliseconds."""
        return float(np.std(self.gaps) * 1000.0) if len(self.gaps) > 1 else 0.0


@dataclass
class Flow:
    key: FlowKey
    start: float
    end: float
    fwd: Direction = field(default_factory=Direction)
    bwd: Direction = field(default_factory=Direction)
    state: str = "INT"
    syn_time: float = 0.0
    synack_time: float = 0.0
    ack_time: float = 0.0
    finished: bool = False

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)

    def handshake(self) -> tuple[float, float, float]:
        """`synack`, `ackdat` and `tcprtt`, in seconds, exactly as UNSW defines them.

        `tcprtt` is the sum of the two halves, not an independent measurement. Returning zeros for
        a flow with no observed handshake is correct: UDP has none, and neither does a TCP flow the
        capture joined mid-conversation.
        """
        synack = self.synack_time - self.syn_time if self.syn_time and self.synack_time else 0.0
        ackdat = self.ack_time - self.synack_time if self.synack_time and self.ack_time else 0.0
        synack = max(synack, 0.0)
        ackdat = max(ackdat, 0.0)
        return synack, ackdat, synack + ackdat


def _require_parser() -> Any:
    try:
        import dpkt
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(
            "pcap parsing needs the `pcap` extra: uv sync --extra pcap. It is optional because "
            "nothing else in Penumbra reads packets."
        ) from exc
    return dpkt


def _ip_to_str(raw: bytes) -> str:
    import socket

    return socket.inet_ntop(socket.AF_INET6 if len(raw) == 16 else socket.AF_INET, raw)


def read_packets(path: Path) -> Any:
    """Yield `(timestamp, src, dst, sport, dport, proto, size, ttl, window, seq, flags)`.

    Non-IP frames are skipped rather than raising: an ARP packet in the capture is normal and is
    not a flow.
    """
    dpkt = _require_parser()
    with open(path, "rb") as handle:  # noqa: PTH123 - dpkt wants a binary file object
        try:
            reader: Any = dpkt.pcapng.Reader(handle)
        except ValueError:
            handle.seek(0)
            reader = dpkt.pcap.Reader(handle)

        for timestamp, buffer in reader:
            try:
                frame = dpkt.ethernet.Ethernet(buffer)
            except Exception:  # noqa: BLE001 - a malformed frame is data, not a bug
                continue
            ip = frame.data
            if not isinstance(ip, dpkt.ip.IP | dpkt.ip6.IP6):
                continue

            ttl = int(getattr(ip, "ttl", getattr(ip, "hlim", 0)))
            payload = ip.data
            sport = dport = 0
            window = seq = 0
            flags = 0
            if isinstance(payload, dpkt.tcp.TCP):
                proto, sport, dport = "tcp", int(payload.sport), int(payload.dport)
                window, seq, flags = int(payload.win), int(payload.seq), int(payload.flags)
            elif isinstance(payload, dpkt.udp.UDP):
                proto, sport, dport = "udp", int(payload.sport), int(payload.dport)
            elif isinstance(payload, dpkt.icmp.ICMP | dpkt.icmp6.ICMP6):
                proto = "icmp"
            else:
                proto = str(getattr(ip, "p", "other"))

            yield (
                float(timestamp),
                _ip_to_str(ip.src),
                _ip_to_str(ip.dst),
                sport,
                dport,
                proto,
                len(buffer),
                ttl,
                window,
                seq,
                flags,
            )


def _tcp_state(flow: Flow, flags: int, dpkt: Any) -> str:
    """UNSW's coarse connection state.

    `FIN` for a closed conversation, `RST` for a reset, `CON` for an established one still running,
    `INT` for a flow with traffic in one direction only — which is what a scan looks like, and is
    the state doing most of the work in the real dataset.
    """
    if flags & dpkt.tcp.TH_RST:
        return "RST"
    if flags & dpkt.tcp.TH_FIN:
        return "FIN"
    if flow.fwd.packets and flow.bwd.packets:
        return "CON"
    return "INT"


def assemble(path: Path | str, *, idle_timeout: float = IDLE_TIMEOUT) -> pd.DataFrame:
    """Turn a capture into a frame of UNSW-NB15-shaped flow features.

    The returned frame carries an `attrs["meta"]` frame of source/destination addresses and start
    times — the same convention the CICIDS loader uses — so correlation and the entity-graph
    features work on live captures too. Addresses never enter the feature matrix.
    """
    dpkt = _require_parser()
    path = Path(path)

    live: dict[FlowKey, Flow] = {}
    done: list[Flow] = []

    for when, src, dst, sport, dport, proto, size, ttl, window, seq, flags in read_packets(path):
        key = FlowKey(src, sport, dst, dport, proto)
        flow = live.get(key)
        forward = True
        if flow is None:
            # Same conversation, opposite direction: the reply belongs to the existing flow, and
            # the endpoint that spoke first stays the source. Missing this splits every
            # conversation into two half-flows with zero bytes in one direction, which is the
            # shape of a scan.
            flow = live.get(key.reverse)
            forward = False

        if flow is not None and (
            when - flow.end > idle_timeout or when - flow.start > ACTIVE_TIMEOUT or flow.finished
        ):
            done.append(flow)
            live.pop(flow.key, None)
            flow = None
            forward = True

        if flow is None:
            flow = Flow(key=key, start=when, end=when)
            live[key] = flow
            forward = True

        side = flow.fwd if forward else flow.bwd
        side.observe(when, size, ttl, window, seq)
        flow.end = max(flow.end, when)

        if proto == "tcp":
            syn = bool(flags & dpkt.tcp.TH_SYN)
            ack = bool(flags & dpkt.tcp.TH_ACK)
            if syn and not ack and not flow.syn_time:
                flow.syn_time = when
            elif syn and ack and not flow.synack_time:
                flow.synack_time = when
            elif ack and flow.synack_time and not flow.ack_time:
                flow.ack_time = when
            flow.state = _tcp_state(flow, flags, dpkt)
            if flags & (dpkt.tcp.TH_FIN | dpkt.tcp.TH_RST):
                flow.finished = True
        else:
            flow.state = "CON" if (flow.fwd.packets and flow.bwd.packets) else "INT"

    done.extend(live.values())
    done.sort(key=lambda f: f.start)
    return _to_frame(done)


def _to_frame(flows: list[Flow]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for flow in flows:
        duration = flow.duration
        safe = duration if duration > 0 else math.nan
        synack, ackdat, tcprtt = flow.handshake()
        fwd, bwd = flow.fwd, flow.bwd

        rows.append(
            {
                "dur": duration,
                "proto": flow.key.proto,
                # Payload-dependent: named in PAYLOAD_DEPENDENT, not guessed from the port.
                "service": "-",
                "state": flow.state,
                "spkts": fwd.packets,
                "dpkts": bwd.packets,
                "sbytes": fwd.bytes,
                "dbytes": bwd.bytes,
                "rate": _safe((fwd.packets + bwd.packets) / safe),
                "sttl": fwd.ttl,
                "dttl": bwd.ttl,
                "sload": _safe(fwd.bytes * 8.0 / safe),
                "dload": _safe(bwd.bytes * 8.0 / safe),
                "sloss": 0,
                "dloss": 0,
                "sinpkt": fwd.mean_gap_ms,
                "dinpkt": bwd.mean_gap_ms,
                "sjit": fwd.jitter_ms,
                "djit": bwd.jitter_ms,
                "swin": fwd.window,
                "stcpb": fwd.tcp_base_seq,
                "dtcpb": bwd.tcp_base_seq,
                "dwin": bwd.window,
                "tcprtt": tcprtt,
                "synack": synack,
                "ackdat": ackdat,
                "smean": fwd.mean_size,
                "dmean": bwd.mean_size,
                "trans_depth": 0,
                "response_body_len": 0,
                "is_ftp_login": 0,
                "ct_ftp_cmd": 0,
                "ct_flw_http_mthd": 0,
                "is_sm_ips_ports": int(flow.key.src == flow.key.dst and flow.key.sport == flow.key.dport),
            }
        )
        meta_rows.append(
            {
                "Timestamp": pd.to_datetime(flow.start, unit="s"),
                "Src IP": flow.key.src,
                "Dst IP": flow.key.dst,
                "Src Port": flow.key.sport,
                "Dst Port": flow.key.dport,
                "Protocol": flow.key.proto,
            }
        )

    frame = pd.DataFrame(rows)
    meta = pd.DataFrame(meta_rows)
    if frame.empty:
        return frame

    _add_connection_counts(frame, meta)
    frame.attrs["meta"] = meta
    frame.attrs["payload_dependent"] = list(PAYLOAD_DEPENDENT)
    frame.attrs["unavailable"] = list(UNAVAILABLE)
    return frame


def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _add_connection_counts(frame: pd.DataFrame, meta: pd.DataFrame) -> None:
    """UNSW's `ct_*` columns: counts over a trailing window of 100 connections.

    Causal by construction — each row counts only the connections *before* it, for the same reason
    `features/entity_graph.py` does. A count that includes the current flow makes the first
    connection of any kind look different from every later one.
    """
    n = len(frame)
    columns = {
        "ct_srv_src": ("Src IP", "Protocol"),
        "ct_srv_dst": ("Dst IP", "Protocol"),
        "ct_dst_ltm": ("Dst IP",),
        "ct_src_ltm": ("Src IP",),
        "ct_src_dport_ltm": ("Src IP", "Dst Port"),
        "ct_dst_sport_ltm": ("Dst IP", "Src Port"),
        "ct_dst_src_ltm": ("Src IP", "Dst IP"),
    }
    keys = {name: list(zip(*(meta[c] for c in cols), strict=True)) for name, cols in columns.items()}

    for name, values in keys.items():
        counts = np.zeros(n, dtype=np.int32)
        for i in range(n):
            window = values[max(0, i - CT_WINDOW) : i]
            counts[i] = window.count(values[i])
        frame[name] = counts

    # ct_state_ttl pairs the connection state with a coarse TTL bucket, which is what makes it a
    # TTL-derived feature and therefore quarantined on UNSW - see data/schema.ARTIFACT_VERDICTS.
    ttl_bucket = pd.cut(frame["sttl"], bins=[-1, 31, 63, 127, 255], labels=[0, 1, 2, 3], include_lowest=True)
    frame["ct_state_ttl"] = frame["state"].astype("category").cat.codes.astype(int) * 4 + ttl_bucket.astype(
        "float"
    ).fillna(0).astype(int)


def summary(frame: pd.DataFrame) -> str:
    """What was assembled, and what could not be."""
    if frame.empty:
        return "  no IP flows in this capture"
    meta = frame.attrs.get("meta", pd.DataFrame())
    span = 0.0
    if len(meta) > 1:
        span = float((meta["Timestamp"].max() - meta["Timestamp"].min()).total_seconds())
    lines = [
        f"  {len(frame):,} flows assembled over {span:,.1f}s",
        f"  {frame['proto'].value_counts().to_dict()}",
        f"  {int(frame['spkts'].sum() + frame['dpkts'].sum()):,} packets, "
        f"{int(frame['sbytes'].sum() + frame['dbytes'].sum()):,} bytes",
        "",
        f"  {len(PAYLOAD_DEPENDENT)} features are zero because computing them needs payload",
        f"  inspection, which Penumbra does not do: {', '.join(PAYLOAD_DEPENDENT)}.",
        f"  {', '.join(UNAVAILABLE)} need TCP sequence reassembly and are also zero.",
        "",
        "  So a pcap-derived flow is NOT identical to a UNSW-NB15 row and the model will score it",
        "  slightly differently. Guessing `service` from the destination port would hide that.",
    ]
    return "\n".join(lines)
