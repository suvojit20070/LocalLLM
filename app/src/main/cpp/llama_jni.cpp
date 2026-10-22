// llama_jni.cpp
// Java <-> llama.cpp bridge for the local OpenAI-compatible server.
//
// - Har request stateless hai (OpenAI style: poori messages list aati hai). Speed ke liye KV cache me
//   pichle prompt ke tokens rakhte hain aur naye prompt ka common prefix reuse karte hain.
// - Gemma 4 ka chat format (<|turn>user ... <turn|>) haath se banta hai, baaki models ke liye
//   llama.cpp ka built-in template.
// - Speed controls: threads (generation/prefill alag), CPU cores pinning (sirf fast cores),
//   batch size, flash attention, KV cache type, mmap on/off, warm-up, benchmark.

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml-cpu.h"

#define TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

enum Format { FMT_GEMMA4, FMT_TEMPLATE };
enum Finish { FINISH_STOP = 0, FINISH_LENGTH = 1, FINISH_CANCELLED = 2 };

struct Session {
    llama_model       *model = nullptr;
    llama_context     *ctx   = nullptr;
    llama_sampler     *smpl  = nullptr;
    const llama_vocab *vocab = nullptr;
    ggml_threadpool   *tp      = nullptr;   // generation threads
    ggml_threadpool   *tpBatch = nullptr;   // prefill threads

    Format      fmt = FMT_TEMPLATE;
    std::string tmpl;
    int         nBatch = 512;

    // load info (modelInfo ke liye)
    int nThreads = 0, nThreadsBatch = 0, flashAttn = -1, kvType = 0, nUbatch = 512;
    bool mmap = true;
    std::string cpuGen, cpuBatch;

    std::vector<llama_token> cached;   // jo tokens abhi KV cache me hain (seq 0)

    // last request stats
    int lastPrompt = 0, lastCompletion = 0, lastReused = 0, lastFinish = FINISH_STOP;
    double lastPrefillMs = 0, lastGenMs = 0;
};

Session           g;
std::mutex        g_mtx;
std::atomic<bool> g_stop{false};
std::atomic<bool> g_loaded{false};   // lock ke bina padh sakte hain
bool              g_backendInit = false;

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---------------- helpers ----------------

void logCallback(ggml_log_level level, const char *text, void *) {
    int prio = ANDROID_LOG_DEBUG;
    if (level == GGML_LOG_LEVEL_ERROR) prio = ANDROID_LOG_ERROR;
    else if (level == GGML_LOG_LEVEL_WARN) prio = ANDROID_LOG_WARN;
    else if (level == GGML_LOG_LEVEL_INFO) prio = ANDROID_LOG_INFO;
    __android_log_print(prio, "llama.cpp", "%s", text);
}

std::string jstr(JNIEnv *env, jstring s) {
    if (!s) return {};
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

std::vector<std::string> jstrArray(JNIEnv *env, jobjectArray arr) {
    std::vector<std::string> out;
    if (!arr) return out;
    jsize n = env->GetArrayLength(arr);
    out.reserve(n);
    for (jsize i = 0; i < n; ++i) {
        auto s = (jstring) env->GetObjectArrayElement(arr, i);
        out.push_back(jstr(env, s));
        if (s) env->DeleteLocalRef(s);
    }
    return out;
}

std::vector<int> jintArr(JNIEnv *env, jintArray arr) {
    std::vector<int> out;
    if (!arr) return out;
    jsize n = env->GetArrayLength(arr);
    out.resize(n);
    if (n > 0) env->GetIntArrayRegion(arr, 0, n, out.data());
    return out;
}

// UTF-8 bytes se Java String (NewStringUTF emoji pe crash karta hai)
jstring toJString(JNIEnv *env, const std::string &s) {
    jbyteArray arr = env->NewByteArray((jsize) s.size());
    env->SetByteArrayRegion(arr, 0, (jsize) s.size(), reinterpret_cast<const jbyte *>(s.data()));
    jclass strCls = env->FindClass("java/lang/String");
    jmethodID ctor = env->GetMethodID(strCls, "<init>", "([BLjava/lang/String;)V");
    jstring charset = env->NewStringUTF("UTF-8");
    auto res = (jstring) env->NewObject(strCls, ctor, arr, charset);
    env->DeleteLocalRef(arr);
    env->DeleteLocalRef(charset);
    env->DeleteLocalRef(strCls);
    return res;
}

std::string trim(const std::string &s) {
    const char *ws = " \t\r\n";
    size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

std::string jsonEscape(const std::string &s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

// Buffer ke end me adhura UTF-8 character ho to usse pehle tak ki length
size_t completeUtf8Prefix(const std::string &s) {
    size_t n = s.size();
    if (n == 0) return 0;
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        auto c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) continue;
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        return (back >= need) ? n : n - back;
    }
    return n;
}

std::vector<llama_token> tokenize(const std::string &text, bool addSpecial) {
    int n = -llama_tokenize(g.vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, addSpecial, true);
    std::vector<llama_token> toks(n > 0 ? n : 0);
    if (n > 0) {
        llama_tokenize(g.vocab, text.c_str(), (int32_t) text.size(), toks.data(), n, addSpecial, true);
    }
    return toks;
}

bool isSystemRole(const std::string &r) { return r == "system" || r == "developer"; }

void clearCache() {
    llama_memory_clear(llama_get_memory(g.ctx), true);
    g.cached.clear();
}

// ---------- Gemma 4 thinking filter ----------
const std::string kOpen  = "<|channel>";
const std::string kClose = "<channel|>";

// Thinking OFF: model ka "<|channel>...<channel|>" ya bina opening ka "...<channel|>" hatao
std::string visibleText(const std::string &raw) {
    std::string s = raw;
    size_t a;
    while ((a = s.find(kOpen)) != std::string::npos) {
        size_t b = s.find(kClose, a + kOpen.size());
        if (b == std::string::npos) { s.erase(a); break; }
        s.erase(a, b + kClose.size() - a);
    }
    size_t c = s.rfind(kClose);
    if (c != std::string::npos) s.erase(0, c + kClose.size());
    size_t b = s.find_first_not_of(" \t\r\n");
    return b == std::string::npos ? std::string() : s.substr(b);
}

// Thinking ON: "<thought>\n...soch...\n</thought>\n\njawab"
std::string thinkingText(const std::string &raw, bool final) {
    std::string out = "<thought>\n";
    size_t c = raw.find(kClose);
    if (c == std::string::npos) {
        out += raw;
        if (final) out = trim(out) + "\n</thought>";
        return out;
    }
    std::string th = raw.substr(0, c);   // trim nahi: stream me bheja text same rehna chahiye
    out += th;
    out += (!th.empty() && th.back() == '\n') ? "</thought>\n\n" : "\n</thought>\n\n";
    out += visibleText(raw.substr(c + kClose.size()));
    return out;
}

// History ke assistant messages se <thought>...</thought> hatao
std::string stripThoughtBlocks(const std::string &in) {
    std::string s = in;
    size_t a;
    while ((a = s.find("<thought>")) != std::string::npos) {
        size_t b = s.find("</thought>", a);
        if (b == std::string::npos) { s.erase(a); break; }
        s.erase(a, b + 10 - a);
    }
    return visibleText(s);
}

// End me adhura tag ("<chan") ho to use abhi mat bhejo
std::string holdPartialTag(const std::string &s) {
    for (const std::string *tag : {&kOpen, &kClose}) {
        for (size_t len = std::min(tag->size() - 1, s.size()); len > 0; --len) {
            if (s.compare(s.size() - len, len, *tag, 0, len) == 0) return s.substr(0, s.size() - len);
        }
    }
    return s;
}

// ---------- prompt formatting ----------

std::string formatGemma4(const std::vector<std::string> &roles, const std::vector<std::string> &contents,
                         bool thinking) {
    std::string out;
    size_t i = 0;
    std::string sys;
    if (!roles.empty() && isSystemRole(roles[0])) {
        sys = trim(contents[0]);
        i = 1;
    }
    if (thinking) {
        out += "<|turn>system\n<|think|>\n" + sys + "<turn|>\n";
    } else if (!sys.empty()) {
        out += "<|turn>system\n" + sys + "<turn|>\n";
    }
    for (; i < roles.size(); ++i) {
        const std::string &r = roles[i];
        if (r == "assistant") {
            out += "<|turn>model\n" + stripThoughtBlocks(contents[i]) + "<turn|>\n";
        } else {
            std::string c = trim(contents[i]);
            if (isSystemRole(r)) c = "[System] " + c;
            out += "<|turn>user\n" + c + "<turn|>\n";
        }
    }
    out += thinking ? "<|turn>model\n<|channel>thought\n" : "<|turn>model\n";
    return out;
}

std::string formatTemplate(const std::vector<std::string> &roles, const std::vector<std::string> &contents) {
    std::vector<std::string> cleaned(contents.size());
    std::vector<llama_chat_message> msgs;
    msgs.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        cleaned[i] = roles[i] == "assistant" ? stripThoughtBlocks(contents[i]) : contents[i];
        const char *role = roles[i] == "developer" ? "system" : roles[i].c_str();
        msgs.push_back({role, cleaned[i].c_str()});
    }
    std::vector<char> buf(16384);
    int n = llama_chat_apply_template(g.tmpl.c_str(), msgs.data(), msgs.size(), true,
                                      buf.data(), (int32_t) buf.size());
    if (n > (int) buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(g.tmpl.c_str(), msgs.data(), msgs.size(), true,
                                      buf.data(), (int32_t) buf.size());
    }
    if (n < 0) return {};
    return std::string(buf.data(), n);
}

// ---------- sampler ----------

struct SampleParams {
    float temp = 0.7f, topP = 0.95f, minP = 0.05f;
    int topK = 40;
    float repeatPenalty = 1.1f, freq = 0.f, presence = 0.f;
    int repeatLastN = 64;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    bool blockThinking = false;
};

void buildSampler(const SampleParams &p) {
    if (g.smpl) { llama_sampler_free(g.smpl); g.smpl = nullptr; }
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    g.smpl = llama_sampler_chain_init(sp);

    // Thinking OFF ho to "<|channel>" token ban, taaki model chhup ke na soche
    if (p.blockThinking) {
        std::vector<llama_logit_bias> bias;
        for (const char *tag : {"<|channel>", "<|think|>"}) {
            std::vector<llama_token> t = tokenize(tag, false);
            if (t.size() == 1) bias.push_back({t[0], -INFINITY});
        }
        if (!bias.empty()) {
            llama_sampler_chain_add(g.smpl, llama_sampler_init_logit_bias(
                    llama_vocab_n_tokens(g.vocab), (int32_t) bias.size(), bias.data()));
        }
    }

    if (p.temp <= 0.0f) {
        if (p.repeatPenalty != 1.0f || p.freq != 0.f || p.presence != 0.f) {
            llama_sampler_chain_add(g.smpl, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(g.smpl, llama_sampler_init_penalties(
                    llama_vocab_n_tokens(g.vocab), p.repeatLastN, p.repeatPenalty, p.freq, p.presence));
        }
        llama_sampler_chain_add(g.smpl, llama_sampler_init_greedy());
        return;
    }

    // top_k pehle: 262k vocab pe penalty lagana slow hai, 40 candidates pe tez
    llama_sampler_chain_add(g.smpl, llama_sampler_init_top_k(p.topK > 0 ? p.topK : 40));
    if (p.repeatPenalty != 1.0f || p.freq != 0.f || p.presence != 0.f) {
        llama_sampler_chain_add(g.smpl, llama_sampler_init_penalties(
                llama_vocab_n_tokens(g.vocab), p.repeatLastN, p.repeatPenalty, p.freq, p.presence));
    }
    llama_sampler_chain_add(g.smpl, llama_sampler_init_top_p(p.topP > 0 ? p.topP : 0.95f, 1));
    if (p.minP > 0) llama_sampler_chain_add(g.smpl, llama_sampler_init_min_p(p.minP, 1));
    llama_sampler_chain_add(g.smpl, llama_sampler_init_temp(p.temp));
    llama_sampler_chain_add(g.smpl, llama_sampler_init_dist(p.seed));
}

// ---------- threads ----------

ggml_threadpool *makePool(int nThreads, const std::vector<int> &cpus) {
    ggml_threadpool_params tpp = ggml_threadpool_params_default(nThreads);
    bool any = false;
    for (int c : cpus) {
        if (c >= 0 && c < GGML_MAX_N_THREADS) { tpp.cpumask[c] = true; any = true; }
    }
    if (!any) return nullptr;   // koi mask nahi -> llama.cpp ka default
    tpp.strict_cpu = false;
    tpp.prio = GGML_SCHED_PRIO_NORMAL;
    return ggml_threadpool_new(&tpp);
}

std::string cpuList(const std::vector<int> &cpus) {
    if (cpus.empty()) return "all";
    std::string s;
    for (int c : cpus) { if (!s.empty()) s += ","; s += std::to_string(c); }
    return s;
}

void unloadAll() {
    g_loaded.store(false);
    if (g.smpl)    { llama_sampler_free(g.smpl);    g.smpl = nullptr; }
    if (g.ctx)     { llama_free(g.ctx);             g.ctx = nullptr; }
    if (g.tp)      { ggml_threadpool_free(g.tp);      g.tp = nullptr; }
    if (g.tpBatch) { ggml_threadpool_free(g.tpBatch); g.tpBatch = nullptr; }
    if (g.model)   { llama_model_free(g.model);     g.model = nullptr; }
    g.vocab = nullptr;
    g.tmpl.clear();
    g.cached.clear();
    g.fmt = FMT_TEMPLATE;
}

ggml_type kvGgmlType(int t) {
    switch (t) {
        case 1:  return GGML_TYPE_Q8_0;
        case 2:  return GGML_TYPE_Q4_0;
        default: return GGML_TYPE_F16;
    }
}

} // namespace

// ======================= JNI =======================

extern "C" {

/**
 * Model load. Success pe null, warna error message.
 * flashAttn: -1 auto, 0 off, 1 on.   kvType: 0 f16, 1 q8_0, 2 q4_0
 * genCpus / batchCpus: kin CPU cores pe threads chalein (khaali = koi bhi)
 */
JNIEXPORT jstring JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeLoad(JNIEnv *env, jclass,
                                              jstring jPath, jint nCtx,
                                              jint nThreads, jint nThreadsBatch,
                                              jint nBatch, jint nUbatch,
                                              jint flashAttn, jint kvType, jboolean useMmap,
                                              jintArray jGenCpus, jintArray jBatchCpus,
                                              jstring jTemplateOverride) {
    std::lock_guard<std::mutex> lock(g_mtx);

    if (!g_backendInit) {
        llama_log_set(logCallback, nullptr);
        llama_backend_init();
        g_backendInit = true;
    }
    unloadAll();

    std::string path = jstr(env, jPath);
    std::vector<int> genCpus = jintArr(env, jGenCpus);
    std::vector<int> batchCpus = jintArr(env, jBatchCpus);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = useMmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;

    g.model = llama_model_load_from_file(path.c_str(), mp);
    if (!g.model) return toJString(env, "Model load fail: " + path);
    g.vocab = llama_model_get_vocab(g.model);

    if (nCtx <= 0) nCtx = std::min(llama_model_n_ctx_train(g.model), 8192);   // 0 = model max (8k tak)
    if (nBatch < 32) nBatch = 32;
    if (nUbatch < 32) nUbatch = 32;
    if (nUbatch > nBatch) nUbatch = nBatch;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) nCtx;
    cp.n_batch         = (uint32_t) nBatch;
    cp.n_ubatch        = (uint32_t) nUbatch;
    cp.n_threads       = nThreads;
    cp.n_threads_batch = nThreadsBatch > 0 ? nThreadsBatch : nThreads;
    cp.no_perf         = true;
    cp.swa_full        = true;     // prefix cache reuse ke liye
    cp.type_k          = kvGgmlType(kvType);
    cp.type_v          = kvGgmlType(kvType);
    // quantized V cache ke liye flash attention zaroori hai
    cp.flash_attn_type = (kvType > 0) ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                      : (llama_flash_attn_type) flashAttn;

    g.ctx = llama_init_from_model(g.model, cp);
    if (!g.ctx && (kvType > 0 || flashAttn == 1)) {
        LOGE("Context fail with FA/KV quant, f16 + auto se dobara try");
        kvType = 0;
        flashAttn = -1;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        g.ctx = llama_init_from_model(g.model, cp);
    }
    if (!g.ctx) {
        unloadAll();
        return toJString(env, "Context create fail (RAM kam hai? context size chhota karo)");
    }

    // Threads ko chune hue cores pe chalao (jaise sirf fast cores)
    g.tp = makePool(nThreads, genCpus);
    g.tpBatch = makePool(cp.n_threads_batch, batchCpus);
    if (g.tp || g.tpBatch) {
        llama_attach_threadpool(g.ctx, g.tp ? g.tp : g.tpBatch, g.tpBatch ? g.tpBatch : g.tp);
    }

    g.nBatch = nBatch;
    g.nUbatch = nUbatch;
    g.nThreads = nThreads;
    g.nThreadsBatch = cp.n_threads_batch;
    g.flashAttn = flashAttn;
    g.kvType = kvType;
    g.mmap = useMmap;
    g.cpuGen = cpuList(genCpus);
    g.cpuBatch = cpuList(batchCpus);

    std::string ov = jstr(env, jTemplateOverride);
    const char *mtc = llama_model_chat_template(g.model, nullptr);
    std::string mt = mtc ? mtc : "";

    if (ov == "gemma4" || (ov.empty() && mt.find("<|turn>") != std::string::npos)) {
        g.fmt = FMT_GEMMA4;
    } else {
        g.fmt = FMT_TEMPLATE;
        g.tmpl = !ov.empty() ? ov : (!mt.empty() ? mt : "chatml");
        llama_chat_message test{"user", "hi"};
        char buf[256];
        if (llama_chat_apply_template(g.tmpl.c_str(), &test, 1, true, buf, sizeof(buf)) < 0) {
            LOGE("Template pehchana nahi gaya, chatml use kar rahe hain");
            g.tmpl = "chatml";
        }
    }

    buildSampler(SampleParams{});

    // Warm-up: pehli request slow na ho (weights page-in + kernels ready)
    {
        std::vector<llama_token> w = tokenize("Hello", true);
        if (!w.empty()) {
            llama_batch b = llama_batch_get_one(w.data(), (int32_t) w.size());
            llama_decode(g.ctx, b);
        }
        clearCache();
    }

    g_loaded.store(true);
    LOGI("Model loaded. ctx=%d threads=%d/%d batch=%d/%d fa=%d kv=%d mmap=%d cpus=%s|%s format=%s",
         nCtx, nThreads, g.nThreadsBatch, nBatch, nUbatch, flashAttn, kvType, (int) useMmap,
         g.cpuGen.c_str(), g.cpuBatch.c_str(), g.fmt == FMT_GEMMA4 ? "gemma4" : "template");
    return nullptr;
}

JNIEXPORT jboolean JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeIsLoaded(JNIEnv *, jclass) {
    return g_loaded.load() ? JNI_TRUE : JNI_FALSE;
}

/** Model + current settings ki jaankari, JSON string */
JNIEXPORT jstring JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeModelInfo(JNIEnv *env, jclass) {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g.model || !g.ctx) return toJString(env, "{}");
    char desc[256] = {0};
    llama_model_desc(g.model, desc, sizeof desc);
    const char *kv[] = {"f16", "q8_0", "q4_0"};
    std::string j = "{";
    j += "\"desc\":\"" + jsonEscape(desc) + "\"";
    j += ",\"n_params\":" + std::to_string(llama_model_n_params(g.model));
    j += ",\"size\":" + std::to_string(llama_model_size(g.model));
    j += ",\"n_ctx_train\":" + std::to_string(llama_model_n_ctx_train(g.model));
    j += ",\"n_ctx\":" + std::to_string(llama_n_ctx(g.ctx));
    j += ",\"n_layer\":" + std::to_string(llama_model_n_layer(g.model));
    j += ",\"n_swa\":" + std::to_string(llama_model_n_swa(g.model));
    j += ",\"n_vocab\":" + std::to_string(llama_vocab_n_tokens(g.vocab));
    j += ",\"format\":\"" + std::string(g.fmt == FMT_GEMMA4 ? "gemma4" : "template") + "\"";
    j += ",\"threads\":" + std::to_string(g.nThreads);
    j += ",\"threads_batch\":" + std::to_string(g.nThreadsBatch);
    j += ",\"batch\":" + std::to_string(g.nBatch);
    j += ",\"ubatch\":" + std::to_string(g.nUbatch);
    j += ",\"flash_attn\":" + std::to_string(g.flashAttn);
    j += ",\"kv_type\":\"" + std::string(kv[std::max(0, std::min(2, g.kvType))]) + "\"";
    j += ",\"mmap\":" + std::string(g.mmap ? "true" : "false");
    j += ",\"cpus_gen\":\"" + g.cpuGen + "\"";
    j += ",\"cpus_batch\":\"" + g.cpuBatch + "\"";
    j += ",\"system_info\":\"" + jsonEscape(llama_print_system_info()) + "\"";
    j += "}";
    return toJString(env, j);
}

/**
 * OpenAI style completion. roles[] / contents[] = poori messages list.
 * Tokens callback.onToken(byte[]) pe stream hote hain (callback null ho sakta hai).
 * Return: poora reply, ya error pe "[ERROR] ..."
 */
JNIEXPORT jstring JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeComplete(JNIEnv *env, jclass,
                                                  jobjectArray jRoles, jobjectArray jContents,
                                                  jint maxTokens, jfloat temp, jfloat topP, jint topK,
                                                  jfloat minP, jfloat repeatPenalty, jint repeatLastN,
                                                  jfloat freqPenalty, jfloat presencePenalty, jint seed,
                                                  jboolean jThinking, jobject callback) {
    std::lock_guard<std::mutex> lock(g_mtx);
    g.lastPrompt = g.lastCompletion = g.lastReused = 0;
    g.lastPrefillMs = g.lastGenMs = 0;
    g.lastFinish = FINISH_STOP;
    if (!g.model || !g.ctx) return toJString(env, "[ERROR] Model loaded nahi hai");

    g_stop.store(false);

    jmethodID onToken = nullptr;
    if (callback) {
        jclass cbCls = env->GetObjectClass(callback);
        onToken = env->GetMethodID(cbCls, "onToken", "([B)V");
        env->DeleteLocalRef(cbCls);
        if (!onToken) return toJString(env, "[ERROR] callback.onToken(byte[]) nahi mila");
    }

    std::vector<std::string> roles = jstrArray(env, jRoles);
    std::vector<std::string> contents = jstrArray(env, jContents);
    if (roles.empty() || roles.size() != contents.size()) {
        return toJString(env, "[ERROR] messages khaali ya galat hain");
    }

    const bool thinking = jThinking == JNI_TRUE && g.fmt == FMT_GEMMA4;
    std::string prompt = (g.fmt == FMT_GEMMA4) ? formatGemma4(roles, contents, thinking)
                                               : formatTemplate(roles, contents);
    if (prompt.empty()) return toJString(env, "[ERROR] Chat template apply nahi hua");

    std::vector<llama_token> toks = tokenize(prompt, true);
    const int nCtx = (int) llama_n_ctx(g.ctx);
    if ((int) toks.size() + 8 > nCtx) {
        return toJString(env, "[ERROR] Prompt bahut lamba hai: " + std::to_string(toks.size()) +
                              " tokens, context size " + std::to_string(nCtx));
    }

    // ---- prefix cache reuse ----
    size_t keep = 0;
    while (keep < g.cached.size() && keep < toks.size() && g.cached[keep] == toks[keep]) ++keep;
    if (keep == toks.size()) --keep;   // last token dobara decode karna hai taaki logits milein

    if (keep < g.cached.size()) {
        llama_memory_t mem = llama_get_memory(g.ctx);
        bool ok = llama_memory_seq_rm(mem, 0, (llama_pos) keep, -1);
        if (!ok || llama_memory_seq_pos_max(mem, 0) != (llama_pos) keep - 1) {
            clearCache();
            keep = 0;
        } else {
            g.cached.resize(keep);
        }
    }
    g.lastReused = (int) keep;
    g.lastPrompt = (int) toks.size();

    double t0 = nowMs();
    for (size_t i = keep; i < toks.size(); i += g.nBatch) {
        int n = (int) std::min((size_t) g.nBatch, toks.size() - i);
        llama_batch batch = llama_batch_get_one(toks.data() + i, n);
        if (llama_decode(g.ctx, batch) != 0) {
            clearCache();
            return toJString(env, "[ERROR] Prompt decode fail");
        }
        g.cached.insert(g.cached.end(), toks.begin() + i, toks.begin() + i + n);
        if (g_stop.load()) {
            g.lastFinish = FINISH_CANCELLED;
            g.lastPrefillMs = nowMs() - t0;
            return toJString(env, "");
        }
    }
    double t1 = nowMs();
    g.lastPrefillMs = t1 - t0;

    SampleParams sp;
    sp.temp = temp; sp.topP = topP; sp.topK = topK; sp.minP = minP;
    sp.repeatPenalty = repeatPenalty > 0 ? repeatPenalty : 1.0f;
    sp.repeatLastN = repeatLastN;
    sp.freq = freqPenalty; sp.presence = presencePenalty;
    sp.seed = seed < 0 ? LLAMA_DEFAULT_SEED : (uint32_t) seed;
    sp.blockThinking = g.fmt == FMT_GEMMA4 && !thinking;
    buildSampler(sp);

    int budget = nCtx - (int) g.cached.size() - 1;
    if (maxTokens <= 0 || maxTokens > budget) maxTokens = budget;

    std::string raw;
    size_t emitted = 0;
    std::string emittedText;
    llama_token tok = 0;
    int generated = 0;
    g.lastFinish = FINISH_LENGTH;

    while (generated < maxTokens) {
        if (g_stop.load()) { g.lastFinish = FINISH_CANCELLED; break; }

        tok = llama_sampler_sample(g.smpl, g.ctx, -1);
        if (llama_vocab_is_eog(g.vocab, tok)) { g.lastFinish = FINISH_STOP; break; }

        char buf[256];
        int n = llama_token_to_piece(g.vocab, tok, buf, sizeof(buf), 0, false);
        if (n < 0) n = 0;
        raw.append(buf, n);

        std::string vis = holdPartialTag(thinking ? thinkingText(raw, false) : visibleText(raw));
        vis.resize(completeUtf8Prefix(vis));
        if (vis.size() > emitted && vis.compare(0, emitted, emittedText) == 0) {
            std::string delta = vis.substr(emitted);
            if (onToken) {
                jbyteArray arr = env->NewByteArray((jsize) delta.size());
                env->SetByteArrayRegion(arr, 0, (jsize) delta.size(), reinterpret_cast<const jbyte *>(delta.data()));
                env->CallVoidMethod(callback, onToken, arr);
                env->DeleteLocalRef(arr);
                if (env->ExceptionCheck()) {
                    env->ExceptionClear();
                    g_stop.store(true);   // client chala gaya
                }
            }
            emittedText = vis;
            emitted = vis.size();
        }

        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(g.ctx, batch) != 0) {
            LOGE("Token decode fail");
            g.lastFinish = FINISH_STOP;
            break;
        }
        g.cached.push_back(tok);
        ++generated;
    }

    g.lastGenMs = nowMs() - t1;
    g.lastCompletion = generated;
    return toJString(env, trim(thinking ? thinkingText(raw, true) : visibleText(raw)));
}

// [promptTokens, completionTokens, reusedTokens, finishReason, prefillMs, genMs]
JNIEXPORT jintArray JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeLastStats(JNIEnv *env, jclass) {
    jint v[6] = {g.lastPrompt, g.lastCompletion, g.lastReused, g.lastFinish,
                 (jint) g.lastPrefillMs, (jint) g.lastGenMs};
    jintArray arr = env->NewIntArray(6);
    env->SetIntArrayRegion(arr, 0, 6, v);
    return arr;
}

/**
 * Speed test: pp tokens prompt process + tg tokens generate.
 * Return [prefill tok/s, generate tok/s], fail pe [-1, -1]. KV cache saaf ho jata hai.
 */
JNIEXPORT jfloatArray JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeBench(JNIEnv *env, jclass, jint pp, jint tg) {
    std::lock_guard<std::mutex> lock(g_mtx);
    jfloat res[2] = {-1.f, -1.f};
    jfloatArray arr = env->NewFloatArray(2);
    if (!g.model || !g.ctx) { env->SetFloatArrayRegion(arr, 0, 2, res); return arr; }

    const int nCtx = (int) llama_n_ctx(g.ctx);
    if (pp + tg + 8 > nCtx) { pp = std::max(16, nCtx / 2); tg = std::min(tg, nCtx / 4); }

    std::vector<llama_token> toks;
    std::string text = "Kolkata is a big city in India with a long history of art, food, music and football. ";
    while ((int) toks.size() < pp) {
        std::vector<llama_token> t = tokenize(text, toks.empty());
        toks.insert(toks.end(), t.begin(), t.end());
    }
    toks.resize(pp);

    clearCache();
    g_stop.store(false);
    double t0 = nowMs();
    for (int i = 0; i < pp; i += g.nBatch) {
        int n = std::min(g.nBatch, pp - i);
        llama_batch b = llama_batch_get_one(toks.data() + i, n);
        if (llama_decode(g.ctx, b) != 0) { clearCache(); env->SetFloatArrayRegion(arr, 0, 2, res); return arr; }
    }
    double t1 = nowMs();

    llama_sampler *greedy = llama_sampler_init_greedy();
    llama_token tok;
    int done = 0;
    for (; done < tg && !g_stop.load(); ++done) {
        tok = llama_sampler_sample(greedy, g.ctx, -1);
        llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(g.ctx, b) != 0) break;
    }
    double t2 = nowMs();
    llama_sampler_free(greedy);
    clearCache();

    res[0] = (float) (pp / ((t1 - t0) / 1000.0));
    res[1] = done > 0 ? (float) (done / ((t2 - t1) / 1000.0)) : -1.f;
    LOGI("bench pp%d=%.1f t/s tg%d=%.1f t/s", pp, res[0], done, res[1]);
    env->SetFloatArrayRegion(arr, 0, 2, res);
    return arr;
}

// Chal rahi generation roko (kisi bhi thread se)
JNIEXPORT void JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeStop(JNIEnv *, jclass) {
    g_stop.store(true);
}

JNIEXPORT void JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeUnload(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mtx);
    unloadAll();
}

JNIEXPORT jstring JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeSystemInfo(JNIEnv *env, jclass) {
    return toJString(env, llama_print_system_info());
}

} // extern "C"
