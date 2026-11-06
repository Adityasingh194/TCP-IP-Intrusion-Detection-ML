# Dataset

**Source:** [Network Intrusion Detection, Kaggle (sampadab17)](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)

The data comes from a simulated US Air Force LAN. Raw TCP/IP dump data was
captured while the network was hit with a range of attacks. Each row is one
**connection**: a sequence of TCP packets that flows between a source IP and a
target IP over a period of time, under a defined protocol. Each connection is
labelled either `normal` or `anomaly` (an attack).

The feature set follows the KDD Cup '99 / NSL-KDD format.
`Train_data.csv` has the same 25,192 rows as the NSL-KDD `KDDTrain+_20Percent` subset.

| File | Rows | Columns | Labels |
|---|---:|---:|---|
| `Train_data.csv` | 25,192 | 42 (41 features + `class`) | `normal` 13,449 / `anomaly` 11,743 |
| `Test_data.csv` | 22,544 | 41 (features only) | **none**. Score it with `predict.py` |

There are no missing values and no duplicate rows. `num_outbound_cmds` and
`is_host_login` are constant in the training data.

## Features

**Categorical (3).** These are one-hot encoded into 80 binary columns.

| Column | Values |
|---|---|
| `protocol_type` | `tcp`, `udp`, `icmp` |
| `service` | 66 network services (`http`, `private`, `ftp_data`, `smtp`, ...) |
| `flag` | 11 connection status flags (`SF` = normal establishment and termination, `S0` = connection attempt with no reply, `REJ` = rejected, ...) |

`Test_data.csv` contains one service, `tftp_u`, that never appears in training. The
encoder ignores it and maps it to all zeros.

**Numeric (38).** These fall into four groups.

| Group | Columns | What they capture |
|---|---|---|
| Basic connection | `duration`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent` | size and shape of a single connection |
| Content | `hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`, `num_outbound_cmds`, `is_host_login`, `is_guest_login` | suspicious behaviour inside the payload (login failures, root access, ...) |
| Time-based traffic (last 2 s) | `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`, `srv_diff_host_rate` | bursts of connections to the same host or service (DoS, probes) |
| Host-based traffic (last 100 connections) | `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate` | slow scans spread over a longer window |
