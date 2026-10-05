"""Context-aware Russian text correction.

Default: ``ai-forever/sage-v1.1.0`` (SAGE 1.7B, FRED-T5 seq2seq). It is trained
for spelling / OCR repair — unlike a chat LLM it replaces non-words using
sentence context instead of rewriting the essay.

Override with ``CORRECTOR_MODEL`` / ``SAGE_MODEL`` (tests use
``sage-fredt5-distilled-95m``). Causal 8B ids such as ``t-tech/T-lite-it-2.1``
still work; on ≤16 GB they load in 4-bit. ``CORRECTOR_QUANT``: ``auto``,
``4bit``, ``8bit``, ``bf16``.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

DEFAULT_MODEL = "ai-forever/sage-v1.1.0"
_TOKENIZER_DEFAULTS = {
    "ai-forever/sage-v1.1.0": "ai-forever/FRED-T5-1.7B",
}
_LM_PREFIX_MODELS = {"ai-forever/sage-v1.1.0"}
_MIN_WEIGHT_BYTES = {
    "t-tech/T-lite-it-2.1": 8_000_000_000,
    "ai-forever/sage-v1.1.0": 4_000_000_000,
    "ai-forever/sage-fredt5-large": 1_000_000_000,
    "ai-forever/sage-fredt5-distilled-95m": 50_000_000,
}
_SYSTEM_PROMPT = (
    "Ты чинишь OCR русской рукописи, а не редактируешь сочинение. "
    "Исправляй только перепутанные буквы, явные опечатки и пунктуацию. "
    "Не перефразируй. Не заменяй слова синонимами. Не улучшай стиль ученика. "
    "Не добавляй слов, которых нет во входе. "
    "Не удлиняй слова. Не меняй род и падеж фамилий. "
    "Крокозябры не превращай в словарные слова, если правка не очевидна. "
    "Если слово неразборчиво — оставь его как есть. "
    "Сохрани абзацы, переносы и порядок. "
    "Верни только исправленный текст, без комментариев."
)
_MISSING_DEPS_MSG = (
    "Контекстный корректор требует torch, transformers, accelerate. "
    "Установите: pip install -r requirements-trocr.txt"
)
log = logging.getLogger(__name__)


def _announce(message: str) -> None:
    print(message, flush=True)
    log.info(message)


def _model_ids() -> tuple[str, str]:
    model_name = (
        os.environ.get("CORRECTOR_MODEL")
        or os.environ.get("SAGE_MODEL")
        or DEFAULT_MODEL
    )
    tokenizer_name = os.environ.get("SAGE_TOKENIZER") or _TOKENIZER_DEFAULTS.get(
        model_name, model_name
    )
    return model_name, tokenizer_name


MODEL_NAME, TOKENIZER_NAME = _model_ids()


def _is_seq2seq(model_name: str) -> bool:
    lowered = model_name.lower()
    return "sage" in lowered or "fredt5" in lowered


def _share_t5_input_embeddings(model) -> None:
    """Point encoder/decoder embed_tokens at ``shared``.

    Passing a custom ``config=`` into ``from_pretrained`` made transformers
    allocate fresh embed_tokens (MISSING in the load report). Generate then
    emitted latin soup because those matrices were random.
    """
    shared = getattr(model, "shared", None)
    if shared is None:
        return
    changed = False
    for block in (getattr(model, "encoder", None), getattr(model, "decoder", None)):
        if block is None or getattr(block, "embed_tokens", None) is None:
            continue
        if block.embed_tokens is not shared:
            block.embed_tokens = shared
            changed = True
    if changed:
        _announce("corrector: tied encoder/decoder embed_tokens to shared")


def _t5_input_embeddings_shared(model) -> bool:
    shared = getattr(model, "shared", None)
    encoder = getattr(model, "encoder", None)
    if shared is None or encoder is None or getattr(encoder, "embed_tokens", None) is None:
        return True
    return encoder.embed_tokens.weight.data_ptr() == shared.weight.data_ptr()


def _sage_num_beams(model_name: str | None = None) -> int:
    """Beam width for SAGE generate.

    ``sage-v1.1.0`` (1.7B) defaults to greedy: 4-beam decode of a ~24-word
    pack can sit in ``generate`` for minutes on a 12 GB card and look hung.
    Smaller SAGE checkpoints keep beam=4. Override with ``SAGE_BEAMS``.
    """
    name = model_name or _model_ids()[0]
    default = "1" if name == "ai-forever/sage-v1.1.0" else "4"
    return max(1, int(os.environ.get("SAGE_BEAMS", default)))


def _checkpoint_bytes(folder: Path) -> int:
    total = 0
    for pattern in ("model.safetensors", "pytorch_model.bin", "model-*.safetensors"):
        for path in folder.glob(pattern):
            if path.is_file():
                total += path.stat().st_size
    return total


def _download_weights(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    min_bytes = _MIN_WEIGHT_BYTES.get(repo_id, 50_000_000)
    _announce(
        f"corrector: fetching {repo_id} "
        f"(need ≥ {min_bytes / 1e9:.1f} GB on disk; Hugging Face progress follows)"
    )
    ignore = ["images/*"]
    if repo_id.startswith("ai-forever/sage-v1.1.0"):
        ignore.extend(["pytorch_model.bin", "pytorch_model*.bin"])
    folder = Path(
        snapshot_download(
            repo_id=repo_id,
            ignore_patterns=ignore,
        )
    )
    size = _checkpoint_bytes(folder)
    if size < min_bytes:
        raise RuntimeError(
            f"Checkpoint {repo_id} on disk is only {size / 1e6:.1f} MB "
            f"(expected ≥ {min_bytes / 1e9:.1f} GB) at {folder}. "
            "Download incomplete. Delete the cache folder and retry."
        )
    _announce(f"corrector: {repo_id} weights {size / 1e9:.2f} GB at {folder}")
    return str(folder)


def _quant_kwargs():
    """Kwargs for from_pretrained. On ≤16 GB GPUs 8B must be 4-bit — never silent bf16."""
    import torch

    mode = os.environ.get("CORRECTOR_QUANT", "auto").lower()
    if not torch.cuda.is_available():
        return {"torch_dtype": torch.float32}
    compute = torch.float16
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    want_4bit = mode in ("4bit", "auto") and vram_gb <= 16 and mode != "bf16"
    if mode == "4bit":
        want_4bit = True
    want_8bit = mode == "8bit"
    if mode == "bf16":
        _announce("corrector: CORRECTOR_QUANT=bf16 (will OOM on 12 GB)")
        return {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
        }
    if want_4bit or want_8bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "На 12 ГБ T-lite 8B нужен 4-bit: pip install bitsandbytes accelerate"
            ) from exc
        cap = "12GiB" if vram_gb <= 16 else f"{int(vram_gb)}GiB"
        if want_4bit:
            _announce(f"corrector: 4-bit NF4 load (GPU {vram_gb:.0f} GB, cap {cap})")
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": {"": 0},
                "max_memory": {0: cap},
                "low_cpu_mem_usage": True,
            }
        _announce(f"corrector: 8-bit load (GPU {vram_gb:.0f} GB, cap {cap})")
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": {"": 0},
            "max_memory": {0: cap},
            "low_cpu_mem_usage": True,
        }
    _announce("corrector: bf16/fp16 full precision")
    return {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }


def _assert_checkpoint(model, model_name: str, seq2seq: bool, n_params: int) -> None:
    """Refuse truncated loads. 4-bit packs two values per byte, so numel is ~half of 8B."""
    cfg = getattr(model, "config", None)
    layers = int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "num_layers", 0) or 0)
    hidden = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "d_model", 0) or 0)
    quantized = bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
    )
    if seq2seq:
        if model_name == "ai-forever/sage-v1.1.0" and n_params < 1_200_000_000:
            raise RuntimeError(
                f"{model_name} has {n_params / 1e6:.0f}M parameters, expected ~1.7B."
            )
        if model_name == "ai-forever/sage-v1.1.0" and not _t5_input_embeddings_shared(model):
            raise RuntimeError(
                "SAGE encoder embeddings are not shared.weight; generate would be garbage. "
                "Do not pass a custom config= into from_pretrained."
            )
        return
    if model_name != "t-tech/T-lite-it-2.1":
        return
    # Qwen3-8B: 36 layers, hidden 4096.
    if layers < 32 or hidden < 3000:
        raise RuntimeError(
            f"{model_name} config looks truncated "
            f"(layers={layers}, hidden={hidden}); expected Qwen3-8B (~36×4096)."
        )
    floor = 3_000_000_000 if quantized else 6_000_000_000
    if n_params < floor:
        raise RuntimeError(
            f"{model_name} has {n_params / 1e6:.0f}M stored tensors "
            f"(floor {floor / 1e9:.1f}B for {'4-bit' if quantized else 'dense'})."
        )


def _loader():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_MSG) from exc

    model_name, tokenizer_name = _model_ids()
    seq2seq = _is_seq2seq(model_name)
    _announce(
        f"corrector: model={model_name} tokenizer={tokenizer_name} "
        f"kind={'seq2seq' if seq2seq else 'causal'} "
        f"(corrector.py={Path(__file__).resolve()})"
    )

    tok_kwargs = {}
    local = _download_weights(model_name)
    tok_source = local if tokenizer_name == model_name else tokenizer_name
    _announce(f"corrector: loading tokenizer {tok_source}")
    tokenizer = AutoTokenizer.from_pretrained(tok_source, **tok_kwargs)
    # FRED-T5 pad is <pad> (id 0), same as decoder_start. Do not alias pad to
    # </s>: that makes T5 start the decoder at EOS and emit punctuation soup.
    if tokenizer.pad_token is None and "FRED-T5" not in tokenizer_name:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    if seq2seq:
        from app import models

        if torch.cuda.is_available():
            models.unload("chandra")
        # Do not pass config= here. A rebuilt T5 config leaves
        # encoder.embed_tokens / decoder.embed_tokens randomly initialized.
        model = AutoModelForSeq2SeqLM.from_pretrained(local, local_files_only=True).eval()
        _share_t5_input_embeddings(model)
        if torch.cuda.is_available():
            model = model.to("cuda")
            _announce("corrector: seq2seq on CUDA")
        else:
            _announce("corrector: seq2seq on CPU")
        # T5 generation_config.max_length defaults to 20. Official SAGE sets
        # max_length from the encoder length; leave room so packs are not
        # truncated to a line of dots.
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = 512
        _announce(f"corrector: SAGE generate num_beams={_sage_num_beams(model_name)}")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "T-lite-it-2.1 (8B) нужен CUDA-GPU. "
                "Для CPU оставьте SAGE: set CORRECTOR_MODEL=ai-forever/sage-fredt5-distilled-95m"
            )
        from app import models

        models.unload("chandra")
        used = models.vram_allocated_gb()
        _announce(f"corrector: VRAM before T-lite load: {used:.2f} GB")
        if used > 1.5:
            _announce("corrector: Chandra still holding VRAM, dropping leftover CUDA modules")
            models.drop_cuda_modules()
            used = models.vram_allocated_gb()
            _announce(f"corrector: VRAM after drop: {used:.2f} GB")
        if used > 4.0:
            raise RuntimeError(
                f"GPU всё ещё занята ({used:.1f} ГБ) после выгрузки Chandra. "
                "Перезапустите uvicorn и повторите: сначала распознавание, затем коррекция."
            )
        model = AutoModelForCausalLM.from_pretrained(
            local, local_files_only=True, **_quant_kwargs()
        ).eval()
        if not getattr(model, "is_loaded_in_4bit", False) and torch.cuda.get_device_properties(0).total_memory / 1e9 <= 16:
            mode = os.environ.get("CORRECTOR_QUANT", "auto").lower()
            hf_q = getattr(model, "hf_quantizer", None)
            quantized = bool(getattr(hf_q, "is_quantized", False) or getattr(model, "is_loaded_in_8bit", False))
            if mode != "bf16" and not quantized:
                raise RuntimeError(
                    "T-lite загрузилась не в 4-bit — на 12 ГБ это даст OOM. "
                    "pip install -U bitsandbytes accelerate transformers"
                )

    n_params = sum(p.numel() for p in model.parameters())
    _assert_checkpoint(model, model_name, seq2seq, n_params)
    quant = (
        "4-bit packed"
        if getattr(model, "is_loaded_in_4bit", False)
        else "8-bit packed"
        if getattr(model, "is_loaded_in_8bit", False)
        else "dense"
    )
    _announce(f"corrector: {n_params / 1e9:.2f}B tensors on device ({quant}) ready")
    return tokenizer, model


def _load():
    from app import models

    return models.load("corrector", _loader)


_HTML_TEXT = re.compile(r"(?<=>)([^<>]+)(?=<)")
_FENCE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)


_CYR = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _max_words() -> int:
    if _is_seq2seq(_model_ids()[0]):
        # SAGE 1.7B is trained on short sentences, not 50–70 word OCR dumps.
        return int(os.environ.get("CORRECTOR_MAX_WORDS", "24"))
    return int(os.environ.get("CORRECTOR_MAX_WORDS", "220"))


def prefer_original(source: str, candidate: str) -> str:
    """Keep OCR when SAGE collapses into punctuation / latin junk."""
    src = source or ""
    dst = (candidate or "").strip()
    if not dst:
        return source
    src_n = len(_CYR.findall(src))
    dst_n = len(_CYR.findall(dst))
    dst_lat = len(_LATIN.findall(dst))
    collapsed = src_n >= 4 and dst_n < max(2, int(src_n * 0.5))
    latin_soup = src_n >= 4 and dst_lat >= 3 and dst_lat >= dst_n
    if collapsed or latin_soup:
        _announce("corrector: SAGE output collapsed, keeping OCR pack")
        return source
    if len(dst) > max(80, int(len(src) * 2.8)):
        _announce("corrector: SAGE output exploded, keeping OCR pack")
        return source
    return candidate


def _line_units(line: str) -> list[str]:
    if not line.strip():
        return [line]
    parts = [p.strip() for p in _SENT_SPLIT.split(line) if p.strip()]
    return parts or [line]


def strip_completion(text: str) -> str:
    """Drop markdown fences / leading labels the chat model sometimes adds."""
    out = text.strip()
    out = _FENCE.sub("", out).strip()
    for prefix in ("Исправленный текст:", "Исправленный текст", "Текст:"):
        if out.lower().startswith(prefix.lower()):
            out = out[len(prefix) :].lstrip(" \n:")
    return out.strip()


def _pack_joiner() -> str:
    # Sentence packs must stay a paragraph. Newlines here became
    # «one sentence per line» in the UI after SAGE.
    return " " if _is_seq2seq(_model_ids()[0]) else "\n"


def _packs(text: str) -> list[str]:
    """Group consecutive sentences so we do a handful of generates, not one per line."""
    limit = _max_words()
    joiner = _pack_joiner()
    packs: list[str] = []
    buf: list[str] = []
    words = 0
    for line in text.split("\n"):
        for unit in _line_units(line):
            w = len(unit.split()) if unit.strip() else 0
            if buf and words + w > limit:
                packs.append(joiner.join(buf))
                buf = [unit]
                words = w
            else:
                buf.append(unit)
                words += w
    if buf:
        packs.append(joiner.join(buf))
    return packs


def apply_to_paragraphs(text: str, fn) -> str:
    """Run ``fn`` on each paragraph; keep original blank lines."""
    parts = re.split(r"(\n+)", text)
    out: list[str] = []
    for part in parts:
        if re.fullmatch(r"\n+", part or "") or not (part and part.strip()):
            out.append(part)
            continue
        pieces = [fn(pack) if pack.strip() else pack for pack in _packs(part)]
        out.append(" ".join(pieces))
    return "".join(out)


def looks_like_form_fragment(text: str) -> bool:
    """Dates, blanks and leftover layout HTML — SAGE treats them as typos."""
    if not text or not text.strip():
        return False
    if "data-bbox" in text or "<div" in text.lower():
        return True
    if "___" in text:
        return True
    if re.search(r"«\s*\d|\d\s*»", text):
        return True
    return False


def _sage(tokenizer, model, text: str) -> str:
    import torch

    model_name, _ = _model_ids()
    payload = ("<LM>" + text) if model_name in _LM_PREFIX_MODELS else text
    inputs = tokenizer(
        payload,
        return_tensors="pt",
        padding="longest",
        truncation=False,
        max_length=None,
    )
    device = getattr(model, "device", None)
    if device is not None:
        inputs = inputs.to(device)
    input_len = int(inputs["input_ids"].size(1))
    # Official sage-v1.1.0 card: max_length = encoder_len * 1.5 (decoder cap).
    # max_new_tokens left T5 at generation_config.max_length=20 → ".л....".
    max_length = max(32, min(512, int(input_len * 1.5)))
    beams = _sage_num_beams(model_name)
    gen_kw: dict = {
        "max_length": max_length,
        "use_cache": True,
        "do_sample": False,
        "num_beams": beams,
    }
    if beams > 1:
        gen_kw["early_stopping"] = True
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kw)
    out = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    if out.startswith("<LM>"):
        out = out[4:]
    return prefer_original(text, out)


def _causal(tokenizer, model, text: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Исправь OCR, не переписывай текст:\n\n{text}"},
    ]
    tmpl = dict(tokenize=False, add_generation_prompt=True)
    try:
        prompt = tokenizer.apply_chat_template(messages, enable_thinking=False, **tmpl)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **tmpl)
    inputs = tokenizer(prompt, return_tensors="pt")
    source = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    inputs = inputs.to(device)
    prompt_len = inputs["input_ids"].shape[1]
    # Output should be ~the same length as the OCR snippet, not 2k of rambling.
    src_len = int(source["input_ids"].shape[1])
    max_new = min(384, max(24, int(src_len * 1.3) + 16))
    eos = tokenizer.eos_token_id
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos,
            pad_token_id=tokenizer.pad_token_id or eos,
        )
    completion_ids = generated[0][prompt_len:]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return strip_completion(raw)


def _correct_one(tokenizer, model, segment: str) -> str:
    if not segment.strip():
        return segment
    if looks_like_form_fragment(segment):
        return segment
    if _is_seq2seq(_model_ids()[0]):
        return _sage(tokenizer, model, segment)
    return _causal(tokenizer, model, segment)


def _correct_segment(tokenizer, model, segment: str) -> str:
    if not segment.strip():
        return segment
    words = segment.split(" ")
    limit = _max_words()
    if len(words) <= limit:
        return _correct_one(tokenizer, model, segment)
    out = [
        _correct_one(tokenizer, model, " ".join(words[i : i + limit]))
        for i in range(0, len(words), limit)
    ]
    return " ".join(out)


def correct_text(text: str, progress=None) -> str:
    """Return a context-corrected version of ``text``.

    Plain text is packed into a few word-bounded chunks (not one generate per
    newline). HTML/markup: only visible text nodes are corrected.
    """
    tokenizer, model = _load()
    from app.cursivefix import fix_cursive_oov

    text = fix_cursive_oov(text)

    if "<" in text and ">" in text:
        nodes = _HTML_TEXT.findall(text)
        done = 0
        total = max(1, len(nodes))

        def _sub(match):
            nonlocal done
            done += 1
            if progress:
                progress(done / total, "correct", token_count=done)
            return _correct_segment(tokenizer, model, match.group(1))

        return fix_cursive_oov(_HTML_TEXT.sub(_sub, text))

    if _is_seq2seq(_model_ids()[0]):
        parts = re.split(r"(\n+)", text)
        blocks = [p for p in parts if p.strip()]
        total = max(1, sum(len(_packs(p)) for p in blocks))
        seen = 0

        def _run(pack: str) -> str:
            nonlocal seen
            seen += 1
            nwords = len(pack.split())
            _announce(f"corrector: pack {seen}/{total} ({nwords} words)")
            if progress:
                progress((seen - 1) / total, "correct", token_count=seen)
            return _correct_one(tokenizer, model, pack) if pack.strip() else pack

        out = apply_to_paragraphs(text, _run)
        if progress:
            progress(1.0, "correct", token_count=total)
        return fix_cursive_oov(out)

    packs = _packs(text)
    total = max(1, len(packs))
    out = []
    for i, pack in enumerate(packs, start=1):
        nwords = len(pack.split())
        _announce(f"corrector: pack {i}/{total} ({nwords} words)")
        if progress:
            progress((i - 1) / total, "correct", token_count=i)
        out.append(_correct_one(tokenizer, model, pack) if pack.strip() else pack)
    if progress:
        progress(1.0, "correct", token_count=total)
    return fix_cursive_oov("\n".join(out))
