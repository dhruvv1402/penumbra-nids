"""Column layouts, label taxonomies, and the specific things each dataset will do to you.

Every constant here was verified against the actual files rather than copied from a paper. The
comments record *why* a column is dropped, because "we dropped six columns" is not a defensible
sentence in a model card and "we dropped `difficulty` because it leaks the answer" is.
"""

from __future__ import annotations

from typing import Final

# =================================================================================================
# NSL-KDD
# =================================================================================================

# The .txt files are headerless. 41 features, then the attack label, then `difficulty`.
NSLKDD_COLUMNS: Final[list[str]] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

NSLKDD_CATEGORICAL: Final[list[str]] = ["protocol_type", "service", "flag"]

# `difficulty` is a LEAK. It encodes how many of 21 classic learners classified the row correctly,
# so it is a function of the answer. Published papers have shipped results with it left in.
#
# `num_outbound_cmds` is identically zero in both splits — no variance, no information.
NSLKDD_DROP: Final[list[str]] = ["difficulty", "num_outbound_cmds"]

# `su_attempted` is documented as binary but the data contains a third value, 2. Left alone it
# becomes a spurious ordinal level. Clamp to {0, 1}.
NSLKDD_CLAMP: Final[dict[str, tuple[int, int]]] = {"su_attempted": (0, 1)}

# 39 attack names across both splits, folded into the four classic categories. The repository's
# own `Attack Types.csv` covers only the 22 training-set names AND uses bare-CR line endings that
# pandas mangles into a single row, so this is hardcoded deliberately.
NSLKDD_ATTACK_CATEGORY: Final[dict[str, str]] = {
    "normal": "normal",
    # --- DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "mailbomb": "dos", "processtable": "dos",
    "udpstorm": "dos",
    # --- Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # --- R2L (remote to local)
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l", "phf": "r2l",
    "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l", "httptunnel": "r2l",
    "named": "r2l", "sendmail": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "worm": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    # --- U2R (user to root)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r", "ps": "u2r",
    "sqlattack": "u2r", "xterm": "u2r",
}

# The 17 attack types present in KDDTest+ but absent from KDDTrain+ — 3,750 rows, 16.6% of the
# test set. The dataset authors built the zero-day experiment for us; this is the ground truth for
# `penumbra eval --unseen-only`.
#
# Consequence for splitting: KDDTrain+ and KDDTest+ must NEVER be concatenated and re-shuffled.
# Doing so destroys the only naturally-occurring unseen-attack holdout in any of our datasets.
NSLKDD_UNSEEN_IN_TEST: Final[frozenset[str]] = frozenset({
    "mscan", "apache2", "processtable", "snmpguess", "saint", "mailbomb", "snmpgetattack",
    "httptunnel", "named", "ps", "sendmail", "xterm", "xlock", "xsnoop", "worm", "udpstorm",
    "sqlattack",
})

# Present in train, absent from test. Noted so the LOAFO harness does not try to score them.
NSLKDD_ABSENT_FROM_TEST: Final[frozenset[str]] = frozenset({"warezclient", "spy"})


# =================================================================================================
# UNSW-NB15  (the pre-split 175k/82k edition)
# =================================================================================================

# Header row is present. 45 columns: id + 42 features + attack_cat + label.
#
# NOTE the absence of srcip / dstip / sport / dsport / stime / ltime. They exist only in the full
# four-part release. This split therefore CANNOT support entity-graph features, per-host sequence
# windows, source-IP alert correlation, or a temporal split. Those all live on CICIDS2017.
UNSW_COLUMNS: Final[list[str]] = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports", "attack_cat", "label",
]

UNSW_CATEGORICAL: Final[list[str]] = ["proto", "service", "state"]

# `id` is a row counter. Keeping it invites the model to memorise ordering.
UNSW_DROP: Final[list[str]] = ["id"]

UNSW_LABEL_BINARY: Final[str] = "label"
UNSW_LABEL_FAMILY: Final[str] = "attack_cat"

UNSW_FAMILIES: Final[list[str]] = [
    "Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
    "Reconnaissance", "Shellcode", "Worms",
]

UNSW_ATTACK_FAMILIES: Final[list[str]] = [f for f in UNSW_FAMILIES if f != "Normal"]

# Training-set counts, verified against the file. Worms at 130 train / 44 test is the imbalance
# story, and 44 test rows is also why Worms recall gets an interval and never a point estimate.
UNSW_TRAIN_COUNTS: Final[dict[str, int]] = {
    "Normal": 56000, "Generic": 40000, "Exploits": 33393, "Fuzzers": 18184, "DoS": 12264,
    "Reconnaissance": 10491, "Analysis": 2000, "Backdoor": 1746, "Shellcode": 1133, "Worms": 130,
}
UNSW_TEST_COUNTS: Final[dict[str, int]] = {
    "Normal": 37000, "Generic": 18871, "Exploits": 11132, "Fuzzers": 6062, "DoS": 4089,
    "Reconnaissance": 3496, "Analysis": 677, "Backdoor": 583, "Shellcode": 378, "Worms": 44,
}

# The `ct_*` columns are already entity-window aggregates computed over a 100-connection window by
# the original Argus/Bro pipeline. Recomputing our own graph features on this dataset would be
# rediscovering what is already in the table — which is why the graph head runs on CICIDS2017.
UNSW_ENTITY_WINDOW_FEATURES: Final[list[str]] = [
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
]

# Suspected testbed artifacts, to be CONFIRMED OR REFUTED by data/audit.py rather than assumed.
# Attack and benign traffic in UNSW-NB15 were generated from different hosts, so time-to-live
# reflects the generator rather than the behaviour. The audit measures each one's solo AUC; only
# features that actually clear ARTIFACT_AUC_THRESHOLD get quarantined.
UNSW_SUSPECTED_ARTIFACTS: Final[list[str]] = ["sttl", "ct_state_ttl", "is_sm_ips_ports", "dttl"]


# =================================================================================================
# CICIDS2017
# =================================================================================================

# Carries Source IP / Destination IP / Timestamp, which is why correlation, graph features, the
# sequence head and the day-based temporal split all live here and not on UNSW-NB15.
CICIDS_ENTITY_COLUMNS: Final[list[str]] = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Timestamp",
]

# `Destination Port` is the worst leak in the dataset: attacks were generated against fixed victim
# ports, so a single decision stump on it scores near-perfectly. Dropped for any honest run; the
# with/without delta is reported to quantify how much it inflates.
CICIDS_LEAK_COLUMNS: Final[list[str]] = ["Destination Port"]

# Identically zero across the corpus — no variance, and they break scalers.
CICIDS_CONSTANT_COLUMNS: Final[list[str]] = [
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

# `Fwd Header Length` appears at BOTH column 35 and column 56 with identical values. pandas
# silently renames the second to `Fwd Header Length.1`; it carries no information.
CICIDS_DUPLICATE_COLUMNS: Final[list[str]] = ["Fwd Header Length.1"]

# Zero-duration flows produce division by zero here.
CICIDS_NONFINITE_PRONE: Final[list[str]] = ["Flow Bytes/s", "Flow Packets/s"]

# Day-based temporal split. Monday is benign-only, so it trains the novelty head; the attack days
# split chronologically. A random split across these files leaks future into past.
CICIDS_TRAIN_DAYS: Final[list[str]] = ["monday", "tuesday", "wednesday"]
CICIDS_TEST_DAYS: Final[list[str]] = ["thursday", "friday"]

# Unlearnable as their own classes at these counts. The evaluation reports them, folded into a
# coarser label, rather than pretending to a per-class recall computed on 11 rows.
CICIDS_ULTRA_RARE: Final[dict[str, int]] = {
    "Heartbleed": 11, "Web Attack - Sql Injection": 21, "Infiltration": 36,
}


# =================================================================================================
# Shared
# =================================================================================================

# The verdict lattice. `priority` orders the queue; it is not a probability. `p_attack` is the only
# calibrated number the system emits. See alerts/models.py.
VERDICTS: Final[tuple[str, ...]] = (
    "KNOWN_ATTACK",      # supervised head recognised a family it was trained on
    "SUSPECTED_NOVEL",   # novelty head fired where the supervised head saw nothing
    "UNCERTAIN",         # conformal set was ambiguous -> routed to human review
    "BENIGN",
    "BENIGN_BY_POLICY",  # matched an analyst-authored suppression rule
)


def nslkdd_category(attack_name: str) -> str:
    """Fold a fine-grained NSL-KDD attack name into {normal, dos, probe, r2l, u2r}.

    Unknown names raise rather than defaulting, because a silent fallback to `normal` would turn a
    parsing bug into a fake benign row.
    """
    key = attack_name.strip().lower().rstrip(".")
    try:
        return NSLKDD_ATTACK_CATEGORY[key]
    except KeyError as exc:
        raise KeyError(f"unmapped NSL-KDD attack name: {attack_name!r}") from exc
