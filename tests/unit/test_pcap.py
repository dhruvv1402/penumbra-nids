"""pcap flow-assembly tests, driven by synthetic captures written packet by packet.

Building the pcap in the test rather than checking in a fixture means every assertion states the
traffic it expects: "two hosts exchange four packets, therefore one bidirectional flow with bytes
in both directions". A binary fixture would make the same assertions unreadable.

The one that matters most is `test_a_reply_joins_the_same_flow`. Keying flows on an unnormalised
5-tuple splits every conversation into two half-flows with zero bytes in one direction — and to a
model trained on real conversations, a one-directional flow is what a scan looks like. The bug
produces no error and turns ordinary web browsing into an attack.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

dpkt = pytest.importorskip("dpkt", reason="pcap assembly needs the optional `pcap` extra")

from penumbra.pcap import assemble as pcap  # noqa: E402


def tcp_packet(
    src: str, dst: str, sport: int, dport: int, *, flags: int = 0, ttl: int = 64, payload: int = 0
) -> bytes:
    import socket

    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, win=8192, seq=1000)
    tcp.data = b"\x00" * payload
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=dpkt.ip.IP_PROTO_TCP, ttl=ttl)
    ip.data = tcp
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(src=b"\xaa" * 6, dst=b"\xbb" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def udp_packet(src: str, dst: str, sport: int, dport: int, *, ttl: int = 64) -> bytes:
    import socket

    udp = dpkt.udp.UDP(sport=sport, dport=dport)
    udp.data = b"\x00" * 16
    udp.ulen = len(udp)
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=dpkt.ip.IP_PROTO_UDP, ttl=ttl)
    ip.data = udp
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(src=b"\xaa" * 6, dst=b"\xbb" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def write_pcap(path: Path, packets: list[tuple[float, bytes]]) -> Path:
    with open(path, "wb") as handle:  # noqa: PTH123
        # Classic little-endian pcap global header, linktype 1 (Ethernet).
        handle.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for when, raw in packets:
            seconds = int(when)
            micros = int(round((when - seconds) * 1_000_000))
            handle.write(struct.pack("<IIII", seconds, micros, len(raw), len(raw)))
            handle.write(raw)
    return path


SYN = dpkt.tcp.TH_SYN
ACK = dpkt.tcp.TH_ACK
FIN = dpkt.tcp.TH_FIN
RST = dpkt.tcp.TH_RST


class TestFlowAssembly:
    def test_a_reply_joins_the_same_flow(self, tmp_path: Path) -> None:
        """The bug that turns web browsing into a scan.

        An unnormalised 5-tuple key makes the reply a separate flow, so both halves have zero bytes
        in one direction - which is exactly the shape of a port scan.
        """
        path = write_pcap(
            tmp_path / "convo.pcap",
            [
                (1.0, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 80, flags=SYN)),
                (1.1, tcp_packet("10.0.0.2", "10.0.0.1", 80, 5000, flags=SYN | ACK)),
                (1.2, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 80, flags=ACK)),
                (1.3, tcp_packet("10.0.0.2", "10.0.0.1", 80, 5000, flags=ACK, payload=400)),
            ],
        )
        frame = pcap.assemble(path)
        assert len(frame) == 1
        assert frame.loc[0, "spkts"] == 2
        assert frame.loc[0, "dpkts"] == 2
        assert frame.loc[0, "dbytes"] > 400

    def test_the_first_speaker_is_the_source(self, tmp_path: Path) -> None:
        path = write_pcap(
            tmp_path / "convo.pcap",
            [
                (1.0, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 80, flags=SYN)),
                (1.1, tcp_packet("10.0.0.2", "10.0.0.1", 80, 5000, flags=SYN | ACK)),
            ],
        )
        meta = pcap.assemble(path).attrs["meta"]
        assert meta.loc[0, "Src IP"] == "10.0.0.1"
        assert meta.loc[0, "Dst Port"] == 80

    def test_a_scan_produces_many_one_directional_flows(self, tmp_path: Path) -> None:
        """And this one SHOULD look like a scan, because it is one."""
        packets = [
            (1.0 + i * 0.01, tcp_packet("10.0.0.9", f"10.0.0.{i + 20}", 40000 + i, 445, flags=SYN))
            for i in range(25)
        ]
        frame = pcap.assemble(write_pcap(tmp_path / "scan.pcap", packets))
        assert len(frame) == 25
        assert (frame["dpkts"] == 0).all()
        assert (frame["state"] == "INT").all()

    def test_an_idle_gap_closes_the_flow(self, tmp_path: Path) -> None:
        path = write_pcap(
            tmp_path / "gap.pcap",
            [
                (1.0, udp_packet("10.0.0.1", "10.0.0.2", 5000, 53)),
                (2.0, udp_packet("10.0.0.1", "10.0.0.2", 5000, 53)),
                # Well past the idle timeout: a new conversation, not packet three of this one.
                (1000.0, udp_packet("10.0.0.1", "10.0.0.2", 5000, 53)),
            ],
        )
        assert len(pcap.assemble(path)) == 2

    def test_fin_closes_the_flow(self, tmp_path: Path) -> None:
        path = write_pcap(
            tmp_path / "fin.pcap",
            [
                (1.0, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 80, flags=SYN)),
                (1.1, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 80, flags=FIN | ACK)),
                (1.2, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 80, flags=SYN)),
            ],
        )
        frame = pcap.assemble(path)
        assert len(frame) == 2
        assert frame.loc[0, "state"] == "FIN"

    def test_a_reset_is_recorded_as_rst(self, tmp_path: Path) -> None:
        path = write_pcap(
            tmp_path / "rst.pcap",
            [
                (1.0, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 81, flags=SYN)),
                (1.05, tcp_packet("10.0.0.2", "10.0.0.1", 81, 5000, flags=RST | ACK)),
            ],
        )
        assert pcap.assemble(path).loc[0, "state"] == "RST"

    def test_non_ip_frames_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        """An ARP packet in a capture is normal and is not a flow."""
        arp = bytes(
            dpkt.ethernet.Ethernet(
                src=b"\xaa" * 6, dst=b"\xff" * 6, type=dpkt.ethernet.ETH_TYPE_ARP, data=dpkt.arp.ARP()
            )
        )
        path = write_pcap(
            tmp_path / "mixed.pcap",
            [(1.0, arp), (1.1, udp_packet("10.0.0.1", "10.0.0.2", 5000, 53))],
        )
        assert len(pcap.assemble(path)) == 1

    def test_an_empty_capture_returns_an_empty_frame(self, tmp_path: Path) -> None:
        frame = pcap.assemble(write_pcap(tmp_path / "empty.pcap", []))
        assert frame.empty
        assert "no IP flows" in pcap.summary(frame)


class TestFeatures:
    @pytest.fixture
    def conversation(self, tmp_path: Path):
        path = write_pcap(
            tmp_path / "f.pcap",
            [
                (10.0, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 80, flags=SYN, ttl=64)),
                (10.2, tcp_packet("10.0.0.2", "10.0.0.1", 80, 5000, flags=SYN | ACK, ttl=128)),
                (10.5, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 80, flags=ACK, ttl=64)),
                (11.0, tcp_packet("10.0.0.2", "10.0.0.1", 80, 5000, flags=ACK, ttl=128, payload=500)),
            ],
        )
        return pcap.assemble(path)

    def test_handshake_timings_match_the_capture(self, conversation) -> None:
        row = conversation.iloc[0]
        assert row["synack"] == pytest.approx(0.2, abs=0.01)
        assert row["ackdat"] == pytest.approx(0.3, abs=0.01)
        # UNSW defines tcprtt as the sum of the two halves, not an independent measurement.
        assert row["tcprtt"] == pytest.approx(row["synack"] + row["ackdat"], abs=1e-6)

    def test_ttl_is_taken_per_direction(self, conversation) -> None:
        """sttl and dttl are properties of the two sending hosts and are not interchangeable."""
        assert conversation.loc[0, "sttl"] == 64
        assert conversation.loc[0, "dttl"] == 128

    def test_derived_features_are_consistent_with_their_primitives(self, conversation) -> None:
        row = conversation.iloc[0]
        assert row["sload"] == pytest.approx(row["sbytes"] * 8.0 / row["dur"], rel=1e-6)
        assert row["rate"] == pytest.approx((row["spkts"] + row["dpkts"]) / row["dur"], rel=1e-6)
        assert row["smean"] == pytest.approx(row["sbytes"] / row["spkts"], rel=1e-6)

    def test_a_zero_duration_flow_does_not_produce_inf(self, tmp_path: Path) -> None:
        """One packet means duration zero, and `sload` would be a division by it."""
        path = write_pcap(tmp_path / "single.pcap", [(1.0, udp_packet("10.0.0.1", "10.0.0.2", 5000, 53))])
        row = pcap.assemble(path).iloc[0]
        assert row["dur"] == 0.0
        assert row["sload"] == 0.0 and row["rate"] == 0.0

    def test_payload_dependent_features_are_zero_and_named(self, conversation) -> None:
        """Guessing `service` from the destination port would hide a real limitation.

        Penumbra inspects flow metadata only (ETHICS_SCOPE.md, MODEL_CARD.md), so these cannot be
        computed here. Reporting them as zero AND listing them is the honest form.
        """
        for feature in pcap.PAYLOAD_DEPENDENT:
            if feature == "service":
                assert conversation.loc[0, feature] == "-"
            else:
                assert conversation.loc[0, feature] == 0
        assert conversation.attrs["payload_dependent"] == list(pcap.PAYLOAD_DEPENDENT)
        assert "payload" in pcap.summary(conversation)


class TestConnectionCounts:
    def test_ct_columns_are_causal(self, tmp_path: Path) -> None:
        """A flow must not be in its own connection count, for the same reason as the graph."""
        packets = [
            (1.0 + i * 0.01, tcp_packet("10.0.0.9", "10.0.0.5", 40000 + i, 445, flags=SYN)) for i in range(10)
        ]
        frame = pcap.assemble(write_pcap(tmp_path / "repeat.pcap", packets))
        assert frame.loc[0, "ct_dst_src_ltm"] == 0
        assert frame.loc[9, "ct_dst_src_ltm"] == 9

    def test_counts_are_bounded_by_the_window(self, tmp_path: Path) -> None:
        """UNSW's ct_* are counts over a trailing 100 connections, not over all history."""
        packets = [
            (1.0 + i * 0.01, tcp_packet("10.0.0.9", "10.0.0.5", 40000 + i, 445, flags=SYN))
            for i in range(150)
        ]
        frame = pcap.assemble(write_pcap(tmp_path / "many.pcap", packets))
        assert frame["ct_dst_src_ltm"].max() <= pcap.CT_WINDOW


class TestSchemaCompatibility:
    def test_every_header_derivable_unsw_feature_is_present(self, tmp_path: Path) -> None:
        """The output has to be scoreable by a model trained on UNSW-NB15, or it is not a converter."""
        from penumbra.data import schema

        path = write_pcap(
            tmp_path / "c.pcap",
            [
                (1.0, tcp_packet("10.0.0.1", "10.0.0.2", 5000, 80, flags=SYN)),
                (1.1, tcp_packet("10.0.0.2", "10.0.0.1", 80, 5000, flags=SYN | ACK)),
            ],
        )
        produced = set(pcap.assemble(path).columns)
        expected = set(schema.UNSW_COLUMNS) - {"id", "attack_cat", "label"}
        assert not expected - produced, f"missing: {sorted(expected - produced)}"

    def test_addresses_never_reach_the_feature_matrix(self, tmp_path: Path) -> None:
        """Entity columns travel in attrs['meta'], as they do for CICIDS. IPs are not features."""
        path = write_pcap(tmp_path / "c.pcap", [(1.0, udp_packet("10.0.0.1", "10.0.0.2", 5000, 53))])
        frame = pcap.assemble(path)
        assert not {"Src IP", "Dst IP", "Timestamp"} & set(frame.columns)
        assert {"Src IP", "Dst IP", "Timestamp"} <= set(frame.attrs["meta"].columns)
