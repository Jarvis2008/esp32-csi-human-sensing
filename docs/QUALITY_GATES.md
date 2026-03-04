# Quality Gates

## Protocol
- Parser handles valid, corrupt, and truncated frames without crashing.

## Replay
- Replay on same input must produce stable state outputs with bounded drift.

## Runtime
- 30-minute runtime: no process crash, bounded packet drop, stable memory trend.

## Inference
- Presence F1 and activity macro-F1 tracked on held-out local sessions.
