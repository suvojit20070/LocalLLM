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

    // ---------- model ----------
    public String getModelPath() { return sp.getString("model_path", ""); }
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

    // ---------- prompt ----------
    public String getSystemPrompt() { return sp.getString("system_prompt", DEFAULT_SYSTEM); }
    public void setSystemPrompt(String s) { sp.edit().putString("system_prompt", s).apply(); }

    // ---------- model settings (badalne pe model reload hota hai) ----------
    /** Context size tokens. 4096 = 8GB phone ke liye safe. 6GB RAM ho to 2048. */
    public int getContextSize() { return sp.getInt("n_ctx", 4096); }
    public void setContextSize(int n) { sp.edit().putInt("n_ctx", n).apply(); }

    public int getThreads() {
        int def = Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors() / 2));
        return sp.getInt("threads", def);
    }
    public void setThreads(int n) { sp.edit().putInt("threads", n).apply(); }

    /** "" = auto. Gemma 4 auto-detect hota hai. */
    public String getTemplateOverride() { return sp.getString("template", ""); }

    // ---------- server ----------
    public int getPort() { return sp.getInt("port", 8080); }
    public void setPort(int p) { sp.edit().putInt("port", p).apply(); }

    /** Request me max_tokens na ho to ye use hota hai */
    public int getDefaultMaxTokens() { return sp.getInt("max_tokens", 1024); }
    public void setDefaultMaxTokens(int n) { sp.edit().putInt("max_tokens", n).apply(); }

    /** Khaali = koi auth nahi. Set ho to "Authorization: Bearer <key>" chahiye. */
    public String getApiKey() { return sp.getString("api_key", ""); }
    public void setApiKey(String k) { sp.edit().putString("api_key", k).apply(); }

    /** false = sirf 127.0.0.1 (isi phone ke apps). true = WiFi network pe bhi. */
    public boolean isLanAccess() { return sp.getBoolean("lan_access", false); }
    public void setLanAccess(boolean b) { sp.edit().putBoolean("lan_access", b).apply(); }

    // ---------- background ----------
    public boolean isAutoStart() { return sp.getBoolean("auto_start", true); }
    public void setAutoStart(boolean b) { sp.edit().putBoolean("auto_start", b).apply(); }

    /** User ne khud Stop dabaya to auto start mat karo jab tak Start na dabaye */
    public boolean isServiceEnabled() { return sp.getBoolean("service_enabled", true); }
    public void setServiceEnabled(boolean b) { sp.edit().putBoolean("service_enabled", b).apply(); }
}
