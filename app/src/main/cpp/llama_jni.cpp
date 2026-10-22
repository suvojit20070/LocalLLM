// llama_jni.cpp
// Java <-> llama.cpp bridge for the local OpenAI-compatible server.
//
// Har request stateless hai (OpenAI style: poori messages list aati hai). Speed ke liye
// KV cache me pichle prompt ke tokens rakhte hain aur naye prompt ka common prefix reuse karte hain,
// isliye same system prompt / same chat history dobara process nahi hoti.
//
// Gemma 4 ka chat format (<|turn>user ... <turn|>) llama_chat_apply_template support nahi karta,
// isliye wo yahan haath se banta hai. Baaki models ke liye llama.cpp ka template use hota hai.

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

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

    Format      fmt = FMT_TEMPLATE;
    std::string tmpl;
    int         nBatch = 512;

    std::vector<llama_token> cached;   // jo tokens abhi KV cache me hain (seq 0)

    // last request stats
    int lastPrompt = 0, lastCompletion = 0, lastReused = 0, lastFinish = FINISH_STOP;
};

Session           g;
std::mutex        g_mtx;
std::atomic<bool> g_stop{false};
std::atomic<bool> g_loaded{false};   // lock ke bina padh sakte hain
bool              g_backendInit = false;

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

// ---------- Gemma 4 thinking filter ----------
// Model kabhi "<|channel>thought ... <channel|>" ya bina opening ke "soch... <channel|>" likh deta hai.
// Client ko sirf final jawab dikhana hai.
const std::string kOpen  = "<|channel>";
const std::string kClose = "<channel|>";

std::string visibleText(const std::string &raw) {
    std::string s = raw;
    size_t a;
    while ((a = s.find(kOpen)) != std::string::npos) {
        size_t b = s.find(kClose, a + kOpen.size());
        if (b == std::string::npos) { s.erase(a); break; }   // thinking abhi chal rahi hai
        s.erase(a, b + kClose.size() - a);
    }
    size_t c = s.rfind(kClose);                               // bina opening ka close tag
    if (c != std::string::npos) s.erase(0, c + kClose.size());
    size_t b = s.find_first_not_of(" \t\r\n");
    return b == std::string::npos ? std::string() : s.substr(b);
}

// Thinking ON: prompt "<|channel>thought\n" pe khatam hota hai, isliye output thinking se shuru hota hai.
// Client ko: "<thought>\n...soch...\n</thought>\n\njawab"
std::string thinkingText(const std::string &raw, bool final) {
    std::string out = "<thought>\n";
    size_t c = raw.find(kClose);
    if (c == std::string::npos) {
        out += raw;
        if (final) out = trim(out) + "\n</thought>";
        return out;
    }
    // trim nahi: stream me pehle bheja gaya text same rehna chahiye
    std::string th = raw.substr(0, c);
    out += th;
    out += (!th.empty() && th.back() == '\n') ? "</thought>\n\n" : "\n</thought>\n\n";
    // jawab me bache hue extra tags hata do
    out += visibleText(raw.substr(c + kClose.size()));
    return out;
}

// History ke assistant messages se <thought>...</thought> hatao (Gemma bhi purani thinking strip karta hai)
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
    // sirf pehla system message system turn banta hai (Gemma 4 template jaisa)
    std::string sys;
    if (!roles.empty() && isSystemRole(roles[0])) {
        sys = trim(contents[0]);
        i = 1;
    }
    if (thinking) {
        out += "<|turn>system\n<|think|>\n" + sys + "<turn|>\n";   // <|think|> = thinking on
    } else if (!sys.empty()) {
        out += "<|turn>system\n" + sys + "<turn|>\n";
    }
    for (; i < roles.size(); ++i) {
        const std::string &r = roles[i];
        if (r == "assistant") {
            out += "<|turn>model\n" + stripThoughtBlocks(contents[i]) + "<turn|>\n";
        } else {
            // user, tool, beech ka system -> user turn
            std::string c = trim(contents[i]);
            if (isSystemRole(r)) c = "[System] " + c;
            out += "<|turn>user\n" + c + "<turn|>\n";
        }
    }
    out += thinking ? "<|turn>model\n<|channel>thought\n"   // seedha thinking se shuru
                    : "<|turn>model\n";                       // thinking off
    return out;
}

std::string formatTemplate(const std::vector<std::string> &roles, const std::vector<std::string> &contents) {
    std::vector<llama_chat_message> msgs;
    msgs.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        const char *role = roles[i] == "developer" ? "system" : roles[i].c_str();
        msgs.push_back({role, contents[i].c_str()});
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

void buildSampler(float temp, float topP, int topK, float freqPenalty, float presencePenalty) {
    if (g.smpl) { llama_sampler_free(g.smpl); g.smpl = nullptr; }
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    g.smpl = llama_sampler_chain_init(sp);

    llama_sampler_chain_add(g.smpl, llama_sampler_init_penalties(
            llama_vocab_n_tokens(g.vocab), 64, 1.1f, freqPenalty, presencePenalty));

    if (temp <= 0.0f) {
        llama_sampler_chain_add(g.smpl, llama_sampler_init_greedy());
        return;
    }
    llama_sampler_chain_add(g.smpl, llama_sampler_init_top_k(topK > 0 ? topK : 40));
    llama_sampler_chain_add(g.smpl, llama_sampler_init_top_p(topP > 0 ? topP : 0.95f, 1));
    llama_sampler_chain_add(g.smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(g.smpl, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(g.smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

void clearCache() {
    llama_memory_clear(llama_get_memory(g.ctx), true);
    g.cached.clear();
}

void unloadAll() {
    g_loaded.store(false);
    if (g.smpl)  { llama_sampler_free(g.smpl); g.smpl = nullptr; }
    if (g.ctx)   { llama_free(g.ctx);          g.ctx = nullptr; }
    if (g.model) { llama_model_free(g.model);  g.model = nullptr; }
    g.vocab = nullptr;
    g.tmpl.clear();
    g.cached.clear();
    g.fmt = FMT_TEMPLATE;
}

} // namespace

// ======================= JNI =======================

extern "C" {

// Model load. Success pe null, warna error message.
// templateOverride: "" = auto, "gemma4", ya llama.cpp template naam ("chatml", "gemma", "llama3"...)
JNIEXPORT jstring JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeLoad(JNIEnv *env, jclass,
                                              jstring jPath, jint nCtx, jint nThreads,
                                              jstring jTemplateOverride) {
    std::lock_guard<std::mutex> lock(g_mtx);

    if (!g_backendInit) {
        llama_log_set(logCallback, nullptr);
        llama_backend_init();
        g_backendInit = true;
    }
    unloadAll();

    std::string path = jstr(env, jPath);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                     // CPU only, phone pe sabse stable
    mp.load_mode    = LLAMA_LOAD_MODE_MMAP;  // file RAM me copy nahi hoti

    g.model = llama_model_load_from_file(path.c_str(), mp);
    if (!g.model) return toJString(env, "Model load fail: " + path);
    g.vocab = llama_model_get_vocab(g.model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) nCtx;
    cp.n_batch         = (uint32_t) g.nBatch;
    cp.n_threads       = nThreads;
    cp.n_threads_batch = nThreads;
    cp.no_perf         = true;
    cp.swa_full        = true;   // prefix cache reuse ke liye zaroori (sliding-window models)

    g.ctx = llama_init_from_model(g.model, cp);
    if (!g.ctx) {
        unloadAll();
        return toJString(env, "Context create fail (RAM kam hai? context size chhota karo)");
    }

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

    buildSampler(0.7f, 0.95f, 40, 0.0f, 0.0f);
    g_loaded.store(true);
    LOGI("Model loaded. ctx=%d threads=%d format=%s", nCtx, nThreads,
         g.fmt == FMT_GEMMA4 ? "gemma4" : "template");
    return nullptr;
}

JNIEXPORT jboolean JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeIsLoaded(JNIEnv *, jclass) {
    return g_loaded.load() ? JNI_TRUE : JNI_FALSE;
}

// OpenAI style completion. roles[] / contents[] = poori messages list.
// Tokens callback.onToken(byte[]) pe stream hote hain (callback null ho sakta hai).
// Return: poora reply, ya error pe "[ERROR] ..."
JNIEXPORT jstring JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeComplete(JNIEnv *env, jclass,
                                                  jobjectArray jRoles, jobjectArray jContents,
                                                  jint maxTokens, jfloat temp, jfloat topP, jint topK,
                                                  jfloat freqPenalty, jfloat presencePenalty,
                                                  jboolean jThinking, jobject callback) {
    std::lock_guard<std::mutex> lock(g_mtx);
    g.lastPrompt = g.lastCompletion = g.lastReused = 0;
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
            return toJString(env, "");
        }
    }

    buildSampler(temp, topP, topK, freqPenalty, presencePenalty);

    int budget = nCtx - (int) g.cached.size() - 1;
    if (maxTokens <= 0 || maxTokens > budget) maxTokens = budget;

    std::string raw;          // model ka poora output (thinking samet)
    size_t emitted = 0;       // visible text ka kitna hissa client ko bhej chuke
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

        // Thinking hata ke jo dikhana hai wahi stream karo
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

    g.lastCompletion = generated;
    return toJString(env, trim(thinking ? thinkingText(raw, true) : visibleText(raw)));
}

// [promptTokens, completionTokens, reusedTokens, finish(0=stop,1=length,2=cancelled)]
JNIEXPORT jintArray JNICALL
Java_com_suvo_localllm_LlamaBridge_nativeLastStats(JNIEnv *env, jclass) {
    jint v[4] = {g.lastPrompt, g.lastCompletion, g.lastReused, g.lastFinish};
    jintArray arr = env->NewIntArray(4);
    env->SetIntArrayRegion(arr, 0, 4, v);
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
