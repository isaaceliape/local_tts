# Phase 1: Core TTS Engine & Model Setup - Research

**Researched:** March 6, 2025  
**Domain:** Local TTS (Kokoro), PyTorch MPS, FastAPI ML Services  
**Confidence:** HIGH

## Summary

Kokoro TTS is a production-ready, open-source text-to-speech model with 82 million parameters that delivers quality comparable to commercial solutions like ElevenLabs. It's specifically designed for efficient local deployment with an Apache 2.0 license. The official Python package (`kokoro` on PyPI) provides a clean KPipeline API that handles text preprocessing, voice management, and audio generation.

For Apple Silicon Macs (M1/M2/M3/M4), Kokoro supports MPS (Metal Performance Shaders) acceleration via PyTorch, though it requires the `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable. The model is lightweight enough to run comfortably on an 8GB MacBook Air M1, with the base model consuming approximately 200-300MB of memory and voice packs adding minimal overhead.

FastAPI provides an ideal backend framework with its lifespan events for singleton model loading, dependency injection for sharing the TTS service across endpoints, and native async support for handling concurrent synthesis requests without blocking.

**Primary recommendation:** Use Kokoro's KPipeline with FastAPI lifespan events for singleton initialization, implement automatic device detection (MPS → CPU fallback), and curate 2-3 high-quality American English voices for the initial release.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| kokoro | >=0.9.4 | TTS inference engine | Official package for Kokoro-82M model, actively maintained |
| misaki[en] | >=0.9.4 | G2P (grapheme-to-phoneme) | Required dependency for English text processing |
| torch | >=2.0 | Deep learning backend | PyTorch with native MPS support for Apple Silicon |
| fastapi | >=0.100 | Web framework | Industry standard for ML APIs, async native |
| uvicorn | >=0.23 | ASGI server | Standard FastAPI server with graceful startup/shutdown |
| soundfile | >=0.12 | Audio I/O | Reliable WAV/MP3 output, handles sample rate conversion |
| espeak-ng | 1.52 | Fallback phonemizer | Handles out-of-vocabulary words, required for OOD fallback |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| loguru | >=0.7 | Logging | Better than stdlib logging for async applications |
| python-multipart | >=0.0.6 | Form parsing | Required by FastAPI for request parsing |
| numpy | >=1.24 | Numerical ops | Often required by torch and soundfile |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Kokoro | Piper TTS | Piper is lighter but lower quality; Kokoro chosen for ElevenLabs-like quality |
| MPS acceleration | CPU only | MPS provides ~2-3x speedup on Apple Silicon; worth the complexity |
| FastAPI | Flask | FastAPI's async support and lifespan events are superior for ML services |
| espeak-ng | No fallback | Without fallback, OOD words are skipped entirely, hurting quality |

**Installation:**
```bash
# macOS (Apple Silicon)
brew install espeak-ng

# Python dependencies
pip install kokoro>=0.9.4 misaki[en] torch fastapi uvicorn soundfile loguru

# For MPS support, set environment variable
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Architecture Patterns

### Recommended Project Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with lifespan
│   ├── config.py            # Settings and device detection
│   ├── dependencies.py      # TTS service dependency injection
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   └── tts_service.py   # Singleton TTS wrapper
│   └── routers/
│       ├── __init__.py
│       └── tts.py           # API endpoints
├── voices/                  # Local voice pack cache (optional)
├── tests/
└── requirements.txt
```

### Pattern 1: FastAPI Lifespan for Singleton Model Loading
**What:** Use FastAPI's `lifespan` context manager to load the TTS model once at startup and share it across all requests.
**When to use:** Always for ML models that are expensive to load (200MB+). Prevents per-request loading delays.

**Example:**
```python
# Source: https://fastapi.tiangolo.com/advanced/events/
from contextlib import asynccontextmanager
from fastapi import FastAPI
from kokoro import KPipeline

# Global state for the TTS service
tts_pipeline: KPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model once
    global tts_pipeline
    device = get_optimal_device()  # mps or cpu
    tts_pipeline = KPipeline(lang_code='a', device=device)
    yield
    # Shutdown: Cleanup
    tts_pipeline = None

app = FastAPI(lifespan=lifespan)

# Dependency injection to access the service
async def get_tts_pipeline() -> KPipeline:
    if tts_pipeline is None:
        raise RuntimeError("TTS pipeline not initialized")
    return tts_pipeline
```

### Pattern 2: Automatic Device Selection with MPS Fallback
**What:** Detect available hardware and automatically select the best device (MPS on Apple Silicon, CPU as fallback).
**When to use:** Required for cross-platform compatibility. MPS requires explicit fallback enablement.

**Example:**
```python
# Source: https://github.com/hexgrad/kokoro & PyTorch MPS docs
import torch
import os

def get_optimal_device() -> str:
    """Auto-select best available device for Apple Silicon."""
    if torch.cuda.is_available():
        return 'cuda'
    
    # MPS requires explicit fallback for some operations
    if (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1' 
        and torch.backends.mps.is_available()):
        return 'mps'
    
    return 'cpu'

# Set this BEFORE importing kokoro or torch in production
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### Pattern 3: Voice Curation and Lazy Loading
**What:** Pre-download curated voice packs at startup, but load voice tensors only when first used.
**When to use:** Voice selection is part of UX; having voices ready prevents first-request latency.

**Example:**
```python
# Source: https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py
CURATED_VOICES = {
    'af_heart': 'American Female - Heart (A grade, highest quality)',
    'af_bella': 'American Female - Bella (A- grade, natural)',
    'am_fenrir': 'American Male - Fenrir (C+ grade, clear)',
}

class TTSService:
    def __init__(self, lang_code: str = 'a'):
        self.pipeline = KPipeline(lang_code=lang_code)
        self.loaded_voices: dict[str, torch.Tensor] = {}
    
    def load_voice(self, voice_id: str) -> torch.Tensor:
        """Lazy load voice tensor, cache for reuse."""
        if voice_id not in self.loaded_voices:
            # KPipeline.load_voice handles HF download if needed
            self.loaded_voices[voice_id] = self.pipeline.load_voice(voice_id)
        return self.loaded_voices[voice_id]
    
    def synthesize(self, text: str, voice_id: str) -> bytes:
        voice = self.load_voice(voice_id)
        # Generate audio...
```

### Pattern 4: Text Preprocessing Pipeline
**What:** Handle numbers, abbreviations, and special characters before sending to TTS.
**When to use:** Kokoro's misaki G2P handles most cases, but preprocessing improves edge cases.

**Key capabilities of misaki G2P:**
- Automatic number expansion (123 → "one hundred twenty-three")
- Abbreviation handling (Dr. → "Doctor")
- Punctuation-aware chunking
- 510 phoneme token limit per chunk (handled automatically by KPipeline)

### Anti-Patterns to Avoid
- **Loading model per request:** Causes 5-10 second delays per request. Use lifespan events.
- **Hardcoding device selection:** Use auto-detection with MPS fallback for portability.
- **Loading all voices upfront:** Voice packs are small but unnecessary; lazy load on first use.
- **Ignoring espeak-ng:** Without it, out-of-vocabulary words are silently skipped.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| G2P (text → phonemes) | Custom regex rules | misaki G2P | Handles numbers, abbreviations, homographs, edge cases |
| Voice management | File-based voice loading | KPipeline.load_voice() | Automatic HF hub download, caching, voice averaging |
| Text chunking | Split by sentences | KPipeline tokenization | Respects 510 phoneme limit, handles punctuation |
| Audio encoding | Raw bytes manipulation | soundfile | Handles WAV headers, sample rates, formats |
| Device detection | Platform sniffing | torch.backends checks | Properly detects MPS, CUDA, CPU availability |

**Key insight:** Kokoro's pipeline handles most complexity internally. Don't intercept or replace its preprocessing unless you have specific requirements.

## Common Pitfalls

### Pitfall 1: MPS Not Enabled Despite Available Hardware
**What goes wrong:** PyTorch on Apple Silicon doesn't use GPU acceleration, falling back to slow CPU inference.
**Why it happens:** MPS fallback must be explicitly enabled via `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable BEFORE importing torch.
**How to avoid:** 
```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Must be before torch import
import torch
from kokoro import KPipeline
```
**Warning signs:** Slow inference (~10x slower than expected), Activity Monitor shows low GPU usage.

### Pitfall 2: espeak-ng Not Installed
**What goes wrong:** Out-of-vocabulary words are silently skipped, producing garbled audio.
**Why it happens:** misaki's G2P can't handle all words; it falls back to espeak-ng for unknown terms.
**How to avoid:** Install espeak-ng at system level before deploying.
```bash
# macOS
brew install espeak-ng

# Verify installation
espeak-ng --version
```
**Warning signs:** Audio skips words, log warnings about "EspeakFallback not Enabled".

### Pitfall 3: Memory Accumulation on Long Texts
**What goes wrong:** Synthesizing very long texts causes memory growth and potential OOM.
**Why it happens:** KPipeline chunks text automatically, but holding all audio chunks in memory at once can exhaust RAM.
**How to avoid:** Stream audio chunks to client, or process in segments. KPipeline yields results as a generator—use it.
```python
# Good: Stream results
for result in pipeline(text, voice='af_heart'):
    yield result.audio

# Bad: Collect all in memory
all_audio = [r.audio for r in pipeline(text, voice)]  # Memory hog
```

### Pitfall 4: Loading Model in Global Scope
**What goes wrong:** Model loads during import, slowing tests and CLI tools that don't need it.
**Why it happens:** Python executes module-level code on import.
**How to avoid:** Use FastAPI lifespan events or lazy initialization patterns.
```python
# Bad: Loads on import
pipeline = KPipeline(lang_code='a')

# Good: Loads on first use or startup event
_pipeline: KPipeline | None = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code='a')
    return _pipeline
```

### Pitfall 5: Incorrect Voice/Language Matching
**What goes wrong:** Using a British voice ('bf_...') with American English pipeline ('a') causes quality degradation.
**Why it happens:** Voice packs are trained for specific language codes and G2P systems.
**How to avoid:** 
- Use `lang_code='a'` for American voices (af_*, am_*)
- Use `lang_code='b'` for British voices (bf_*, bm_*)
- Check voice prefix matches pipeline language code

## Code Examples

### Complete FastAPI TTS Endpoint

```python
# Source: Pattern from FastAPI docs + Kokoro README
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from kokoro import KPipeline
import torch
import io
import soundfile as sf
import os

# Ensure MPS fallback is enabled
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Global state
_tts_service: KPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TTS model once at startup."""
    global _tts_service
    
    # Device selection
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"Using MPS device for acceleration")
    else:
        print(f"MPS not available, using CPU")
    
    # Initialize pipeline
    _tts_service = KPipeline(lang_code='a', device=device)
    print("TTS model loaded successfully")
    
    yield
    
    # Cleanup
    _tts_service = None
    print("TTS service shutdown")

app = FastAPI(lifespan=lifespan)

async def get_tts() -> KPipeline:
    if _tts_service is None:
        raise HTTPException(503, "TTS service not available")
    return _tts_service

@app.post("/synthesize")
async def synthesize(
    text: str,
    voice: str = "af_heart",
    tts: KPipeline = Depends(get_tts)
):
    """Synthesize text to speech."""
    if not text.strip():
        raise HTTPException(400, "Text cannot be empty")
    
    try:
        # Generate audio
        results = list(tts(text, voice=voice))
        
        if not results:
            raise HTTPException(400, "No audio generated")
        
        # Concatenate audio chunks
        audio = torch.cat([r.audio for r in results if r.audio is not None])
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio.numpy(), 24000, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    
    except Exception as e:
        raise HTTPException(500, f"Synthesis failed: {str(e)}")
```

### Device Detection Utility

```python
# Source: PyTorch MPS docs + Kokoro implementation
import torch
import os
import warnings

def setup_device() -> str:
    """
    Detect and configure optimal device for inference.
    Returns: 'cuda', 'mps', or 'cpu'
    """
    # Check CUDA
    if torch.cuda.is_available():
        return 'cuda'
    
    # Check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1':
            warnings.warn(
                "MPS available but PYTORCH_ENABLE_MPS_FALLBACK not set. "
                "Set it before importing torch for GPU acceleration."
            )
            return 'cpu'
        return 'mps'
    
    return 'cpu'

def get_device_info() -> dict:
    """Get detailed device information for debugging."""
    return {
        'device': setup_device(),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'mps_built': torch.backends.mps.is_built(),
        'mps_fallback_enabled': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1',
    }
```

### Voice Curation Configuration

```python
# Source: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
CURATED_VOICES = {
    # American Female - Highest quality
    'af_heart': {
        'name': 'Heart',
        'gender': 'female',
        'language': 'en-US',
        'quality': 'A',
        'description': 'Warm, natural American female voice. Best overall quality.',
        'recommended': True,
    },
    # American Female - Alternative
    'af_bella': {
        'name': 'Bella', 
        'gender': 'female',
        'language': 'en-US',
        'quality': 'A-',
        'description': 'Clear, professional American female voice.',
        'recommended': True,
    },
    # American Male
    'am_fenrir': {
        'name': 'Fenrir',
        'gender': 'male',
        'language': 'en-US',
        'quality': 'C+',
        'description': 'Clear American male voice. Good for variety.',
        'recommended': True,
    },
}

def get_recommended_voices() -> list[dict]:
    """Get list of curated voices for UI dropdown."""
    return [
        {
            'id': voice_id,
            'display_name': f"{info['name']} ({info['gender']}, {info['quality']})",
            'description': info['description'],
        }
        for voice_id, info in CURATED_VOICES.items()
        if info['recommended']
    ]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Piper TTS (lightweight) | Kokoro (quality-focused) | 2024 | Kokoro offers ElevenLabs-level quality at 82M params |
| Startup/shutdown events | Lifespan context managers | FastAPI 0.95+ | Cleaner resource management, ASGI-native |
| Manual phoneme conversion | misaki G2P library | 2024 | Integrated with Kokoro, handles edge cases |
| CUDA-only GPU accel | MPS for Apple Silicon | PyTorch 1.12+ | Mac users get GPU acceleration without CUDA |
| Per-request model loading | Singleton with lifespan | Always | Essential for sub-second response times |

**Deprecated/outdated:**
- `espeak` (old): Use `espeak-ng` instead (actively maintained)
- `@app.on_event("startup")`: Use `lifespan` parameter instead (modern FastAPI)
- Manual WAV encoding: Use `soundfile` library (handles edge cases)

## Open Questions

### 1. **MPS Performance on M1 8GB**
- What we know: MPS provides acceleration but may have higher memory overhead than CPU
- What's unclear: Exact performance delta on 8GB vs 16GB MacBook Air
- Recommendation: Implement device benchmarking on startup, allow manual override via env var

### 2. **Voice License for Distribution**
- What we know: Kokoro model is Apache 2.0, voice packs are part of the model
- What's unclear: Whether pre-downloading voices in installer has any restrictions
- Recommendation: Let Kokoro's HF hub download handle voice fetching (lazy, on-demand)

### 3. **Long Text Handling**
- What we know: KPipeline auto-chunks at 510 phoneme limit (~400 chars)
- What's unclear: Optimal chunk size for natural prosody vs memory usage
- Recommendation: Start with default chunking, optimize if users report issues

### 4. **Offline Operation**
- What we know: Kokoro downloads model from HF hub on first use
- What's unclear: Best practice for pre-downloading in packaged app
- Recommendation: Implement `huggingface_hub.snapshot_download()` in setup script

## Sources

### Primary (HIGH confidence)
- [PyPI: kokoro 0.9.4](https://pypi.org/project/kokoro/) - Official package documentation
- [GitHub: hexgrad/kokoro](https://github.com/hexgrad/kokoro) - Source code, pipeline implementation
- [HuggingFace: Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - Model card, voice documentation
- [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/) - Official pattern for startup/shutdown
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html) - Official MPS documentation

### Secondary (MEDIUM confidence)
- [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) - Voice quality grades and recommendations
- [Homebrew: espeak-ng](https://formulae.brew.sh/formula/espeak-ng) - macOS installation instructions
- [GitHub: hexgrad/misaki](https://github.com/hexgrad/misaki) - G2P library documentation

### Tertiary (LOW confidence)
- Community Discord: [https://discord.gg/QuGxSWBfQy](https://discord.gg/QuGxSWBfQy) - Real-world deployment experiences (not verified)
- Inference benchmarks: Various blog posts showing M1 performance (methodology unclear)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official PyPI packages, actively maintained
- Architecture: HIGH - FastAPI lifespan is well-documented, Kokoro API is stable
- Pitfalls: HIGH - MPS requirement is documented, espeak-ng requirement is clear
- Performance: MEDIUM - Exact M1 8GB metrics not officially benchmarked

**Research date:** March 6, 2025  
**Valid until:** June 6, 2025 (90 days - stable stack but fast-moving field)
