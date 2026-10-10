# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Exact distinct counts of every column of every Parquet file in a directory.

usage: truth.py <dir> <out.json> <benchmark> <sf> <generator>
Output: {"benchmark", "scale_factor", "generator", "ndv": {table: {column: n}}}
Counts exclude NULLs (SQL COUNT(DISTINCT)); computed exactly with DuckDB.
"""
import duckdb, glob, json, os, sys
d, out, bench, sf, gen = sys.argv[1:6]
con = duckdb.connect()
con.execute("SET threads TO 2; SET memory_limit='5GB'; SET preserve_insertion_order=false")
ndv = {}
for path in sorted(glob.glob(os.path.join(d, "*.parquet")) + glob.glob(os.path.join(d, "*/"))):
    table = os.path.basename(path.rstrip("/")).removesuffix(".parquet")
    if table == "dbgen_version":
        continue
    src = path if path.endswith(".parquet") else os.path.join(path, "*.parquet")
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src}')").fetchall()]
    row = con.execute("SELECT " + ", ".join(f'COUNT(DISTINCT "{c}")' for c in cols)
                      + f" FROM read_parquet('{src}')").fetchone()
    ndv[table] = dict(zip(cols, row))
    print(table, len(cols), file=sys.stderr)
json.dump({"benchmark": bench, "scale_factor": float(sf) if "." in sf else int(sf),
           "generator": gen, "ndv": ndv}, open(out, "w"), indent=1, sort_keys=False)
