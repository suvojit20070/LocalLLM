package com.suvo.localllm;

import android.content.Context;
import android.content.SharedPreferences;

import java.io.File;

/** Saari settings ek jagah. */
public final class Prefs {

    public static final String MODEL_URL =
            "https://huggingface.co/NobodyWho/Google_Gemma4-E2B-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf";
    public static final String MODEL_FILE = "gemma-4-E2B-it-Q4_K_M.gguf";

    public static final String DEFAULT_SYSTEM =
            "You are a helpful assistant. Reply in the same language the user uses (Hindi, English or Hinglish). Keep answers short and clear.";

    public static final String PRESET_ROCKET = "rocket";
    public static final String PRESET_BALANCED = "balanced";
    public static final String PRESET_QUALITY = "quality";

    private static final String FILE = "llm_prefs";
    private final SharedPreferences sp;

    public Prefs(Context ctx) {
        sp = ctx.getApplicationContext().getSharedPreferences(FILE, Context.MODE_PRIVATE);
    }

    /** Model files yahan rehti hain (app ka private folder, koi permission nahi chahiye). */
    public static File modelDir(Context ctx) {
        File d = ctx.getExternalFilesDir("models");
        if (d == null) d = new File(ctx.getFilesDir(), "models");
        //noinspection ResultOfMethodCallIgnored
        d.mkdirs();
        return d;
    }

    private int i(String k, int def) { return sp.getInt(k, def); }
    private float f(String k, float def) { return sp.getFloat(k, def); }
    private boolean b(String k, boolean def) { return sp.getBoolean(k, def); }
    private String s(String k, String def) { return sp.getString(k, def); }

    // ================= model =================
    public String getModelPath() { return s("model_path", ""); }
    public void setModelPath(String p) { sp.edit().putString("model_path", p).apply(); }

    public boolean hasModel() {
        String p = getModelPath();
        return !p.isEmpty() && new File(p).isFile();
    }

    /** /v1/models me dikhne wala naam */
    public String getModelId() {
        String p = getModelPath();
        if (p.isEmpty()) return "local-model";
        String n = new File(p).getName();
        return n.toLowerCase().endsWith(".gguf") ? n.substring(0, n.length() - 5) : n;
    }

    public long getDownloadId() { return sp.getLong("download_id", -1); }
    public void setDownloadId(long id) { sp.edit().putLong("download_id", id).apply(); }

    // ================= prompt =================
    public String getSystemPrompt() { return s("system_prompt", DEFAULT_SYSTEM); }
    public void setSystemPrompt(String v) { sp.edit().putString("system_prompt", v).apply(); }

    // ================= load settings (badalne pe model reload) =================
    /** Context size tokens. 0 = model ka max (8192 tak). */
    public int getContextSize() { return i("n_ctx", 4096); }
    /** Generation threads */
    public int getThreads() { return i("threads", CpuInfo.get().recommendedThreads()); }
    /** Prompt padhne (prefill) ke threads */
    public int getThreadsBatch() { return i("threads_batch", CpuInfo.get().recommendedBatchThreads()); }
    public int getBatch() { return i("batch", 512); }
    public int getUbatch() { return i("ubatch", 512); }
    /** -1 auto, 0 off, 1 on */
    public int getFlashAttn() { return i("flash_attn", -1); }
    /** 0 f16, 1 q8_0, 2 q4_0 */
    public int getKvType() { return i("kv_type", 0); }
    public boolean isMmap() { return b("mmap", true); }
    /** Generation threads sirf fast cores pe */
    public boolean isFastCoresGen() { return b("fast_gen", true); }
    /** Prefill threads sirf fast cores pe */
    public boolean isFastCoresBatch() { return b("fast_batch", false); }
    /** "" = auto. Gemma 4 auto-detect hota hai. */
    public String getTemplateOverride() { return s("template", ""); }

    public void saveLoadSettings(int nCtx, int threads, int threadsBatch, int batch, int ubatch,
                                 int flashAttn, int kvType, boolean mmap,
                                 boolean fastGen, boolean fastBatch) {
        sp.edit()
                .putInt("n_ctx", nCtx)
                .putInt("threads", threads)
                .putInt("threads_batch", threadsBatch)
                .putInt("batch", batch)
                .putInt("ubatch", ubatch)
                .putInt("flash_attn", flashAttn)
                .putInt("kv_type", kvType)
                .putBoolean("mmap", mmap)
                .putBoolean("fast_gen", fastGen)
                .putBoolean("fast_batch", fastBatch)
                .apply();
    }

    /** Load settings ka ek fingerprint: badla to model reload karna hai */
    public String loadKey() {
        return getModelPath() + "|" + getContextSize() + "|" + getThreads() + "|" + getThreadsBatch()
                + "|" + getBatch() + "|" + getUbatch() + "|" + getFlashAttn() + "|" + getKvType()
                + "|" + isMmap() + "|" + isFastCoresGen() + "|" + isFastCoresBatch() + "|" + getTemplateOverride();
    }

    // ================= generation defaults (request me na ho to) =================
    public int getDefaultMaxTokens() { return i("max_tokens", 1024); }
    public float getTemperature() { return f("temperature", 0.7f); }
    public float getTopP() { return f("top_p", 0.95f); }
    public int getTopK() { return i("top_k", 40); }
    public float getMinP() { return f("min_p", 0.05f); }
    public float getRepeatPenalty() { return f("repeat_penalty", 1.1f); }
    public int getRepeatLastN() { return i("repeat_last_n", 64); }
    /** Thinking on: reply ke content me pehle <thought>...</thought>, phir jawab (sirf Gemma 4) */
    public boolean isThinking() { return b("thinking", true); }

    public void saveGenSettings(int maxTokens, float temp, float topP, int topK, float minP,
                                float repeatPenalty, int repeatLastN, boolean thinking) {
        sp.edit()
                .putInt("max_tokens", maxTokens)
                .putFloat("temperature", temp)
                .putFloat("top_p", topP)
                .putInt("top_k", topK)
                .putFloat("min_p", minP)
                .putFloat("repeat_penalty", repeatPenalty)
                .putInt("repeat_last_n", repeatLastN)
                .putBoolean("thinking", thinking)
                .apply();
    }

    public void setThinking(boolean v) { sp.edit().putBoolean("thinking", v).apply(); }

    // ================= self learning (memory) =================
    public boolean isLearning() { return b("learning", true); }
    public void setLearning(boolean v) { sp.edit().putBoolean("learning", v).apply(); }
    /** Seekhi hui baatein system prompt me daalo */
    public boolean isUseMemory() { return b("use_memory", true); }
    public void setUseMemory(boolean v) { sp.edit().putBoolean("use_memory", v).apply(); }
    public int getMaxMemories() { return i("max_memories", 40); }
    public void setMaxMemories(int v) { sp.edit().putInt("max_memories", v).apply(); }
    /** Kitne second khaali rehne ke baad seekhna shuru ho */
    public int getLearnIdleSeconds() { return i("learn_idle", 20); }

    // ================= server =================
    public int getPort() { return i("port", 8080); }
    public void setPort(int p) { sp.edit().putInt("port", p).apply(); }

    /** Khaali = koi auth nahi. Set ho to "Authorization: Bearer <key>" chahiye. */
    public String getApiKey() { return s("api_key", ""); }
    public void setApiKey(String k) { sp.edit().putString("api_key", k).apply(); }

    /** false = sirf 127.0.0.1 (isi phone ke apps). true = WiFi network pe bhi. */
    public boolean isLanAccess() { return b("lan_access", false); }
    public void setLanAccess(boolean v) { sp.edit().putBoolean("lan_access", v).apply(); }

    // ================= background =================
    public boolean isAutoStart() { return b("auto_start", true); }
    public void setAutoStart(boolean v) { sp.edit().putBoolean("auto_start", v).apply(); }

    /** User ne khud Stop dabaya to auto start mat karo jab tak Start na dabaye */
    public boolean isServiceEnabled() { return b("service_enabled", true); }
    public void setServiceEnabled(boolean v) { sp.edit().putBoolean("service_enabled", v).apply(); }

    // ================= presets =================

    /** Ek click me poori setting. Rocket = sabse tez, Quality = sabse achha jawab. */
    public void applyPreset(String preset) {
        CpuInfo cpu = CpuInfo.get();
        int fast = cpu.recommendedThreads();
        int all = cpu.recommendedBatchThreads();
        switch (preset) {
            case PRESET_ROCKET:
                saveLoadSettings(2048, fast, all, 512, 512, 1, 1, true, true, false);
                saveGenSettings(256, 0.6f, 0.9f, 20, 0.05f, 1.1f, 64, false);
                break;
            case PRESET_QUALITY:
                saveLoadSettings(8192, fast, all, 512, 512, -1, 0, true, true, false);
                saveGenSettings(2048, 0.7f, 0.95f, 40, 0.05f, 1.05f, 128, true);
                break;
            default: // balanced
                saveLoadSettings(4096, fast, all, 512, 512, -1, 0, true, true, false);
                saveGenSettings(1024, 0.7f, 0.95f, 40, 0.05f, 1.1f, 64, isThinking());
        }
    }
}
