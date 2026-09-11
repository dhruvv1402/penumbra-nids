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
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
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
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "mailbomb": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    # --- Probe
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "mscan": "probe",
    "saint": "probe",
    # --- R2L (remote to local)
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "httptunnel": "r2l",
    "named": "r2l",
    "sendmail": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "worm": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    # --- U2R (user to root)
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
}

# The 17 attack types present in KDDTest+ but absent from KDDTrain+ — 3,750 rows, 16.6% of the
# test set. The dataset authors built the zero-day experiment for us; this is the ground truth for
# `penumbra eval --unseen-only`.
#
# Consequence for splitting: KDDTrain+ and KDDTest+ must NEVER be concatenated and re-shuffled.
# Doing so destroys the only naturally-occurring unseen-attack holdout in any of our datasets.
NSLKDD_UNSEEN_IN_TEST: Final[frozenset[str]] = frozenset(
    {
        "mscan",
        "apache2",
        "processtable",
        "snmpguess",
        "saint",
        "mailbomb",
        "snmpgetattack",
        "httptunnel",
        "named",
        "ps",
        "sendmail",
        "xterm",
        "xlock",
        "xsnoop",
        "worm",
        "udpstorm",
        "sqlattack",
    }
)

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
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "attack_cat",
    "label",
]

UNSW_CATEGORICAL: Final[list[str]] = ["proto", "service", "state"]

# `id` is a row counter. Keeping it invites the model to memorise ordering.
UNSW_DROP: Final[list[str]] = ["id"]

UNSW_LABEL_BINARY: Final[str] = "label"
UNSW_LABEL_FAMILY: Final[str] = "attack_cat"

UNSW_FAMILIES: Final[list[str]] = [
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

UNSW_ATTACK_FAMILIES: Final[list[str]] = [f for f in UNSW_FAMILIES if f != "Normal"]

# Training-set counts, verified against the file. Worms at 130 train / 44 test is the imbalance
# story, and 44 test rows is also why Worms recall gets an interval and never a point estimate.
UNSW_TRAIN_COUNTS: Final[dict[str, int]] = {
    "Normal": 56000,
    "Generic": 40000,
    "Exploits": 33393,
    "Fuzzers": 18184,
    "DoS": 12264,
    "Reconnaissance": 10491,
    "Analysis": 2000,
    "Backdoor": 1746,
    "Shellcode": 1133,
    "Worms": 130,
}
UNSW_TEST_COUNTS: Final[dict[str, int]] = {
    "Normal": 37000,
    "Generic": 18871,
    "Exploits": 11132,
    "Fuzzers": 6062,
    "DoS": 4089,
    "Reconnaissance": 3496,
    "Analysis": 677,
    "Backdoor": 583,
    "Shellcode": 378,
    "Worms": 44,
}

# The `ct_*` columns are already entity-window aggregates computed over a 100-connection window by
# the original Argus/Bro pipeline. Recomputing our own graph features on this dataset would be
# rediscovering what is already in the table — which is why the graph head runs on CICIDS2017.
UNSW_ENTITY_WINDOW_FEATURES: Final[list[str]] = [
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
]

# Suspected testbed artifacts, CONFIRMED OR REFUTED by data/audit.py rather than assumed.
# See ARTIFACT_VERDICTS below for what the audit actually found.
UNSW_SUSPECTED_ARTIFACTS: Final[list[str]] = ["sttl", "ct_state_ttl", "is_sm_ips_ports", "dttl"]


# =================================================================================================
# Artifact verdicts
# =================================================================================================
#
# `data/audit.py` SURFACES candidates. It cannot decide whether a highly discriminative feature is a
# testbed artifact or genuine attack behaviour, because that is a question about network semantics
# rather than about statistics.
#
# The distinction matters in both directions. Quarantining a real signal cripples the model for no
# reason; keeping an artifact inflates every number we report. So each candidate gets an explicit
# verdict with a stated reason, and only `artifact` verdicts are excluded from the headline run.
#
# The test for "artifact": is there a causal path from attack behaviour to this feature value? A
# SYN-flood genuinely produces a SYN error rate of 1.0. Nothing about an attack causes a packet's
# initial TTL to be 31 - that is the host that generated it.

Verdict = str  # "artifact" | "signal" | "unresolved"

ARTIFACT_VERDICTS: Final[dict[str, dict[str, tuple[Verdict, str]]]] = {
    "unsw": {
        "sttl": (
            "artifact",
            "sttl=31 is 22.5% of train rows and 100% benign; sttl=62/254 dominate attacks. "
            "Initial TTL is a property of the sending host's OS, not of attack behaviour. "
            "UNSW-NB15 generated attack and benign traffic from different machines.",
        ),
        "dttl": (
            "artifact",
            "dttl=29 is 22.5% of train rows and 100% benign. Same reasoning as sttl, on the "
            "destination side.",
        ),
        "is_sm_ips_ports": (
            "artifact",
            "is_sm_ips_ports=1 is 100% benign. Flags source IP/port equal to destination IP/port, "
            "which is a testbed addressing quirk rather than a defensive signal.",
        ),
        "proto": (
            "artifact",
            "proto='unas' is 6.9% of train rows and 100% attack. 'unas' means the generator did "
            "not assign a protocol - it labels the synthesis process, not the traffic.",
        ),
        "ct_state_ttl": (
            "artifact",
            "Highest solo AUC in the dataset (0.86) and derived from the same TTL values as sttl "
            "and dttl. Excluded with them; keeping a TTL-derived aggregate after removing TTL "
            "would leave the artifact in through the back door.",
        ),
        # The ct_* window counters saturate at 16-18 and 33 and those exact values are ~100%
        # attack. This is genuinely ambiguous: high connection counts to one destination ARE what
        # scanning looks like, but the specific saturation values look like pipeline behaviour.
        # Reported both ways rather than decided by assertion.
        "ct_dst_sport_ltm": ("unresolved", "Window counter saturating at 16-18, ~100% attack."),
        "ct_src_dport_ltm": ("unresolved", "Window counter saturating at 16-17, ~99% attack."),
        "ct_dst_ltm": ("unresolved", "Window counter saturating at 17/33, ~99% attack."),
        "ct_dst_src_ltm": ("unresolved", "Window counter saturating at 16-17, ~99% attack."),
        "ct_src_ltm": ("unresolved", "Window counter, leaky at saturation values."),
        "ct_srv_src": ("unresolved", "Window counter, leaky at saturation values."),
    },
    "nslkdd": {
        # NSL-KDD's leaky values are overwhelmingly REAL BEHAVIOUR. A SYN flood produces a SYN
        # error rate of exactly 1.0 by definition; that is the detection working, not a leak.
        # Quarantining these would remove the signal the dataset exists to carry.
        "srv_serror_rate": (
            "signal",
            "srv_serror_rate=1.0 is 99% attack because a SYN flood IS a 100% SYN-error condition. "
            "Causal, not incidental.",
        ),
        "dst_host_srv_serror_rate": ("signal", "Same SYN-error semantics, host-aggregated."),
        "dst_host_serror_rate": ("signal", "Same SYN-error semantics, host-aggregated."),
        "serror_rate": ("signal", "Same SYN-error semantics."),
        "diff_srv_rate": (
            "signal",
            "A high rate of connections to differing services is what a port sweep is.",
        ),
        "same_srv_rate": ("signal", "Complement of diff_srv_rate; same reasoning."),
        "srv_diff_host_rate": ("signal", "Connections to many hosts on one service - scanning."),
        "dst_host_srv_diff_host_rate": ("signal", "Host-aggregated scanning behaviour."),
        "dst_host_srv_count": ("signal", "Connection-count aggregate; behavioural."),
        "count": ("signal", "Connection count in the time window; behavioural."),
        "service": (
            "signal",
            "service='domain_u' is 99.9% benign because DNS lookups genuinely are mostly benign "
            "in this capture. A prior over service, not a generator fingerprint.",
        ),
    },
}


# Some verdicts are value-level, not column-level. `proto` is a legitimate feature - TCP versus UDP
# is real network behaviour - and only the generator's `unas` marker is the artifact. Quarantining
# the whole column to remove one value would throw away a genuine signal to fix a synthetic one.
#
# This matters most for rule mining, where the unit of exclusion is a condition rather than a
# column: `proto=unas > 0.5` must go and `proto=tcp > 0.5` must stay.
ARTIFACT_VALUES: Final[dict[str, dict[str, list[str]]]] = {
    "unsw": {"proto": ["unas"]},
}


def artifact_verdict(dataset: str, feature: str) -> tuple[Verdict, str]:
    """Verdict for one feature. Unknown features are `unresolved`, never silently `signal`."""
    return ARTIFACT_VERDICTS.get(dataset, {}).get(feature, ("unresolved", "No recorded judgment."))


def quarantined(dataset: str) -> set[str]:
    """Every feature whose recorded verdict is `artifact`.

    Value-level quarantines come back as indicator names (`proto=unas`) rather than as the whole
    column, so a consumer excluding this set removes the artifact and keeps the feature.

    `unresolved` features are deliberately NOT in here. The ct_* window counters are genuinely
    ambiguous - high connection counts to one destination are what scanning looks like - and
    quarantining on suspicion would be the same error as keeping on convenience, in the other
    direction.
    """
    key = dataset.lower().replace("-", "").replace("_", "")
    values = ARTIFACT_VALUES.get(key, {})
    out: set[str] = set()
    for feature, (verdict, _) in ARTIFACT_VERDICTS.get(key, {}).items():
        if verdict != "artifact":
            continue
        if feature in values:
            out.update(f"{feature}={v}" for v in values[feature])
        else:
            out.add(feature)
    return out


# =================================================================================================
# CICIDS2017
# =================================================================================================

# Carries Source IP / Destination IP / Timestamp, which is why correlation, graph features, the
# sequence head and the day-based temporal split all live here and not on UNSW-NB15.
CICIDS_ENTITY_COLUMNS: Final[list[str]] = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
]

# `Destination Port` is the worst leak in the dataset: attacks were generated against fixed victim
# ports, so a single decision stump on it scores near-perfectly. Dropped for any honest run; the
# with/without delta is reported to quantify how much it inflates.
CICIDS_LEAK_COLUMNS: Final[list[str]] = ["Destination Port"]

# Identically zero across the corpus — no variance, and they break scalers.
CICIDS_CONSTANT_COLUMNS: Final[list[str]] = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
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
    "Heartbleed": 11,
    "Web Attack - Sql Injection": 21,
    "Infiltration": 36,
}


# =================================================================================================
# Shared
# =================================================================================================

# The verdict lattice. `priority` orders the queue; it is not a probability. `p_attack` is the only
# calibrated number the system emits. See alerts/models.py.
VERDICTS: Final[tuple[str, ...]] = (
    "KNOWN_ATTACK",  # supervised head recognised a family it was trained on
    "SUSPECTED_NOVEL",  # novelty head fired where the supervised head saw nothing
    "UNCERTAIN",  # conformal set was ambiguous -> routed to human review
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
