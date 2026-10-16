"""The chat-model client every script uses: DeepSeek's models through OpenRouter when OPENROUTER_API_KEY is set,
else DeepSeek's own API (DEEPSEEK_API_KEY).

Callers keep DeepSeek's names ("deepseek-flash", "deepseek-v4-pro") and its thinking switch
(extra_body={"thinking": {"type": "disabled"}}); on OpenRouter both are translated here, and the model's
reasoning comes back as `reasoning_content`, as on DeepSeek. The names also stay the labels stored with
LLM output (SessionSummaries.model, ChunkContexts.model), so switching provider never invalidates them.
"""
import os

from dotenv import dotenv_values
from openai import AsyncOpenAI, OpenAI

from common import REPO_ROOT

OPENROUTER_MODELS = {"deepseek-flash": "~deepseek/deepseek-flash-latest",
                     "deepseek-v4-pro": "~deepseek/deepseek-pro-latest"}


def _key(name):
    return os.environ.get(name) or dotenv_values(REPO_ROOT / ".env").get(name)


def available():
    return bool(_key("OPENROUTER_API_KEY") or _key("DEEPSEEK_API_KEY"))


def provider():
    return "openrouter" if _key("OPENROUTER_API_KEY") else "deepseek"


def client(asynchronous=False, timeout=180):
    """An OpenAI-style client (only .chat.completions.create is used), or None when no key is set."""
    cls = AsyncOpenAI if asynchronous else OpenAI
    if _key("OPENROUTER_API_KEY"):
        return _OpenRouter(cls(api_key=_key("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1",
                               timeout=timeout), asynchronous)
    if _key("DEEPSEEK_API_KEY"):
        return cls(api_key=_key("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=timeout)
    return None


def _translate(kw):
    kw = dict(kw)
    kw["model"] = OPENROUTER_MODELS.get(kw["model"], kw["model"])
    extra = dict(kw.pop("extra_body", None) or {})
    thinking = extra.pop("thinking", {}).get("type") != "disabled"
    extra["reasoning"] = {"enabled": thinking}
    extra["provider"] = {"require_parameters": True}   # only providers that honour tools / JSON mode
    kw["extra_body"] = extra
    # DeepSeek wants its reasoning passed back within a tool loop as reasoning_content; OpenRouter calls it reasoning
    kw["messages"] = [{("reasoning" if k == "reasoning_content" else k): v for k, v in m.items()}
                      if isinstance(m, dict) and "reasoning_content" in m else m for m in kw["messages"]]
    return kw


def _expose_reasoning(obj):
    if obj is not None and getattr(obj, "reasoning_content", None) is None:
        obj.reasoning_content = getattr(obj, "reasoning", None)
    return obj


def _fix(resp):
    if hasattr(resp, "choices"):
        for c in resp.choices or []:
            _expose_reasoning(getattr(c, "message", None) or getattr(c, "delta", None))
        u = resp.usage
        if u is not None and getattr(u, "prompt_cache_hit_tokens", None) is None:
            u.prompt_cache_hit_tokens = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return resp


class _OpenRouter:
    def __init__(self, inner, asynchronous):
        self.chat = self
        self.completions = self
        self._inner, self._async = inner, asynchronous

    def create(self, **kw):
        kw = _translate(kw)
        if self._async:
            return self._acreate(kw)
        r = self._inner.chat.completions.create(**kw)
        return (_fix(ch) for ch in r) if kw.get("stream") else _fix(r)

    async def _acreate(self, kw):
        return _fix(await self._inner.chat.completions.create(**kw))
