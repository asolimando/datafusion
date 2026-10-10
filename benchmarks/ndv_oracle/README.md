<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# True distinct counts of the benchmark data

Exact `COUNT(DISTINCT col)` (NULLs excluded) of every column of the TPC-H and
TPC-DS tables, used as the reference for distinct count estimates
(`--ndv-oracle <file>` in `dfbench tpch` / `dfbench tpcds`).

| File               | Data                                                       |
| ------------------ | ---------------------------------------------------------- |
| `tpch_sf1.json`    | `tpchgen-cli` 3.0.0, scale factor 1                        |
| `tpch_sf10.json`   | `tpchgen-cli` 3.0.0, scale factor 10                       |
| `tpcds_sf1.json`   | the prebuilt `tpcds/data/sf1` files of `apache/datafusion-benchmarks`, as downloaded by `bench.sh data tpcds` |
| `tpcds_sf10.json`  | `tpcgen-cli` (dsdgen 2.0.0), scale factor 10               |

The generators are deterministic: the counts depend only on the generator
version and the scale factor, not on how the data is written (number of files,
row groups, page size, compression or writer). Regenerating `tpchgen-cli` SF1
and SF10 tables, `tpchgen-cli --parts 4`, and `tpcgen-cli` SF10 tables gave
the same rows. Counts do not scale from one scale factor to another: keys grow
with it, small domains do not.

Each file records its generator; when the generator is upgraded, regenerate
the data and the file:

```shell
python truth.py <directory of the parquet files> tpch_sf1.json tpch 1 "tpchgen-cli 3.0.0"
```

`truth.py` needs `duckdb` (`pip install duckdb`).
