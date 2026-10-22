# Local LLM Server (Android, Java + llama.cpp)

Phone pe Gemma 4 E2B (ya koi bhi GGUF model) chalata hai aur ek **local OpenAI-compatible endpoint** deta hai.
App me koi chat UI nahi hai: sirf model, system prompt aur settings.

```
Base URL : http://127.0.0.1:8080/v1
Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (stream bhi)
```

## Build (PC pe Android Studio)

1. llama.cpp source clone karo:
   ```
   git clone --depth 1 https://github.com/ggml-org/llama.cpp app/src/main/cpp/llama.cpp
   ```
2. SDK Manager > SDK Tools: **NDK 27.x** aur **CMake 3.22.1** install karo.
3. Gradle sync, phir Run. Pehli native build 5-10 min legi.

## Pehli baar

1. App kholo. Model nahi hai to **Download Gemma 4 E2B (3.1 GB)** dabao, ya **Phone se .gguf file chuno**.
2. Model aate hi server apne aap start hota hai (notification dikhegi).
3. **Battery optimization off karo** wala button dabao.

Iske baad app band kar do. Server background me chalta rahega, phone restart pe bhi wapas start hoga.

## Use example

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="anything")
print(c.chat.completions.create(model="gemma",
      messages=[{"role":"user","content":"Hi"}]).choices[0].message.content)
```

Termux se bhi same URL chalega.

## Naye features

- **Speed preset:** 🚀 Rocket / ⚖️ Balanced / 🧠 Quality, phir Save & apply.
- **Speed test:** prefill aur generate tok/s naapta hai. Settings badal ke compare karo.
- **Performance settings:** context (0 = model max), threads (jawab / prompt alag), batch, ubatch,
  flash attention, KV cache f16/q8/q4, fast cores pinning, mmap.
- **Self-learning:** khaali time me chats se user ke baare me baatein seekh ke memory me rakhta hai,
  har request ke system prompt me jodta hai. `GET/POST/DELETE /v1/memory`.
  Request me `"learn": false` ya `"memory": false` bhej sakte ho.
- **`GET /v1/info`:** model, settings, last request timing.
- Har reply me `timings` (llama.cpp server jaisa): prompt/generate tok/s.
- `scripts/chat.py`, `scripts/loop_chat.py`: Termux test scripts.

## Notes

- Request me system message na ho to app wala default system prompt lagta hai.
- Supported params: `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`,
  `frequency_penalty`, `presence_penalty`, `stream`, `stream_options.include_usage`.
- Tools / function calling aur images abhi supported nahi hain.
- Ek time pe ek reply banta hai, baaki requests line me wait karti hain.
- Same system prompt / chat history ka prefix KV cache se reuse hota hai, isliye multi-turn chat fast rehta hai.
- API key set karoge to `Authorization: Bearer <key>` bhejna hoga.
- LAN switch on karne pe same WiFi ke doosre devices bhi use kar sakte hain.
- 6 GB RAM phone: context size 2048 karo. Phone zyada garam ho to threads 2-3.
- Purane phone pe app turant crash (SIGILL) ho to `CMakeLists.txt` me `GGML_CPU_ARM_ARCH` line comment karo.
