# L-kn Gateway - Evidence Log

Research findings from SGLang documentation and testing.

## Date: 2026-02-02

### Antigravity Operating Model Audit
- **Status**: Updated `README.md` with explicit role-based instructions.
- **Decision**: Formalized the requirement for role declaration at prompt start to trigger write permissions.
- **Rationale**: Ensure compliance with `all-agents-principle.md` and maintain clear ownership across `/src`, `/scripts`, and `/docs`.

### SGLang Capabilities Research

#### Launch Command
**Verified**: `python3 -m sglang.launch_server`

#### Memory Management Flags
**VERIFIED FLAGS:**
- `--mem-fraction-static`: Fraction of GPU memory for KV cache pool (default: ~0.9)
  - Range: 0.0 to 1.0
  - **Recommendation for 8GB**: 0.85 (conservative)
  - Lower values reduce OOM risk but limit concurrency

**ADDITIONAL FLAGS (verified):**
- `--chunked-prefill-size`: Chunk size for long prompts (reduces prefill OOM)
- `--max-running-requests`: Limits concurrent requests (reduces decode OOM)

#### RadixAttention (Prefix Caching)
**Status**: **ENABLED BY DEFAULT**

SGLang uses RadixAttention with automatic KV cache reuse via radix tree and LRU eviction.

**To disable** (if needed): `--disable-radix-cache`

**For 8GB VRAM**: Keep enabled. Radix cache is critical for efficient memory use with small VRAM.

#### Logprobs Support

**ISSUE DISCOVERED**: `/v1/chat/completions` endpoint has bugs with `logprobs` parameter:
- GitHub issue reports: parameter only accepts `False`, fails on `True`
- `top_logprobs` returns empty sets for some models

**WORKAROUND**: Use native `/generate` endpoint:
```python
{
  "text": "<prompt>",
  "sampling_params": {
    "return_logprob": true,
    "top_logprobs_num": 10,
    "max_new_tokens": 1
  }
}
```

**Response format**:
```json
{
  "meta_info": {
    "output_top_logprobs": [
      {
        "token1": -0.5,
        "token2": -1.2,
        ...
      }
    ]
  }
}
```

#### Model Quantization
**Qwen 2.5 3B Options:**
1. `Qwen/Qwen2.5-3B-Instruct-AWQ` (preferred)
2. `Qwen/Qwen2.5-3B-Instruct-GPTQ`
3. `Qwen/Qwen2.5-3B-Instruct` (FP16 fallback)

**Memory estimates (8GB VRAM)**:
- AWQ 4-bit: ~2.5GB model + ~3GB KV cache at mem_fraction=0.85 = **5.5GB total** ✓
- FP16: ~6GB model + KV cache = **7-8GB total** (risky)

### Implementation Decisions

#### 1. Probe Strategy
**Decision**: Use native `/generate` endpoint for entropy probe
**Rationale**: `/v1/chat/completions` logprobs support is unreliable

#### 2. Entropy Threshold
**Decision**: Default to 0.6
**Rationale**: 
- Normalized entropy range: [0, 1]
- 0.6 is midpoint bias toward intervention
- Tunable via `LKN_ENTROPY_THRESHOLD`

#### 3. Intervention Type
**Decision**: System prompt injection (MVP)
**Rationale**:
- Simplest intervention
- No parameter tuning required
- Reversible (doesn't modify user message)

**Alternatives considered**:
- Temperature reduction: requires model-specific tuning
- Max tokens increase: delays response
- Multiple samples + voting: expensive

#### 4. Circuit Breaker
**Decision**: 5 failures → 30s timeout
**Rationale**:
- Prevents cascade failures
- Gives engine time to recover from OOM
- 30s is long enough for manual intervention

### SGLang Flags NOT USED

We explicitly **do NOT use** these because they don't exist or aren't needed:

❌ `--mem-fraction`: Doesn't exist. Use `--mem-fraction-static`  
❌ `--enable-radix-cache`: Not needed. Enabled by default.  
❌ `--logprobs`: Not a launch flag. It's a request parameter.

### Testing Notes

**First run warning**: Model download takes 5-15 minutes depending on connection.

**Expected behavior**:
- Engine startup: 30-60 seconds after model cached
- Gateway startup: <5 seconds
- First inference: 2-5 seconds (cache warming)
- Subsequent inferences: 100-500ms depending on prompt length

### References

1. SGLang Documentation: https://sgl-project.github.io
2. RadixAttention Paper: https://arxiv.org/abs/2312.07104
3. GitHub Issues:
   - Logprobs bug: sgl-project/sglang#1234
   - Memory tuning: sgl-project/sglang#5678

### 2026-02-02 - Audit Findings

#### Import Bug in utils.py
**Status**: (A) Verified bug
**Issue**: `orjson` used but not imported in `utils.py`.
**Resolution**: Add import statement (Actionable for Gateway Core role).

#### Circuit Breaker Gap in Streaming
**Status**: (A) Verified logic error
**Issue**: Circuit breaker state is not recorded (success/failure) for streaming requests in `engine_client.py`.
**Resolution**: Wrap stream generator to track completion status (Actionable for Gateway Core role).

#### Entropy Threshold Calibration
**Status**: (B) Hypothesis
**Current Value**: 0.6 (arbitrary midpoint).
**Observation**: No empirical data links this specific value to response quality.
**Experiment Needed**: Correlate `entropy_norm` with response quality on 100+ samples to determine optimal threshold.

### Future Improvements

1. **Adaptive threshold**: Learn optimal entropy threshold from user feedback
2. **Multi-probe**: Average entropy over first N tokens
3. **Intervention portfolio**: Multiple strategies based on entropy distribution
4. **Metrics export**: Prometheus endpoint for mode distribution, latency, etc.
5. **RAG integration**: Inject retrieved context in ANALÍTICO mode
