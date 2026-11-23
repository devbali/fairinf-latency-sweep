# PD KV migration time per request (gpt-oss-20b, NIXL over NVLink)

`requests.csv.gz` holds what `src/estimators/pd_kv_transfer.py` is fitted on. It has one row for each of the 58,058
probe requests of the `pd_load.py` runs `gptoss-pd-xfer-instr-nopoll-nixl-20261111` and
`gptoss-pd-xfer-instr-rest-nixl-20261111`. Both runs use fork 99f39f2b5, 1 prefill + 1 decode A100, and the
instrumentation hooks of `src/pd_transfer/instrument/`.

| column | meaning |
|---|---|
| load | the background load while the probe ran |
| fold | cross-validation fold, by probe batch |
| place | the request's 0-based position in its prefill batch's send order |
| reqs_left | requests of the batch still to send, this one included |
| pieces_left | sum over them of min(KV write pieces, 32) + state write pieces |
| MB_left | their KV bytes, MB |
| d_next_ms | the next prefill step's duration, from the 20b eager prefill model (degree 2 + sum l^2 of `gptoss-clean-20261028`), floor 89.4 ms, plus two cached-context terms for chunk continuations |
| n_ov | sends overlapping the next step: min(reqs_left, max(0, 50 (d_next_ms - 82) / d_next_ms - place)) |
| migration_ms | measured: prefill_finished_time to the decode server's last NIXL notification for the request |

The model:

```
T = 1.864 + 2.197 reqs_left + 0.1428 pieces_left + 0.0264 MB_left + n_ov x (0.01891 D_next - 1.867)   ms
```

Held out by probe batch (5 folds), its absolute error is 1.48 ms at the median, 6.07 ms at p90 and 21.6 ms at p99.
It came from pruning a 12-term model, whose error was 1.49 / 5.88 / 22.5 ms. The terms dropped were the overlapped
pieces, the last request's pieces, the pieces over 32 and the decode server's state. Each removal cost at most
0.2 ms at p90.
