package com.suvo.localllm;

/**
 * llama.cpp ka Java wrapper (JNI).
 * load / complete / unload sab ek hi background thread se call karo (LlmService ka worker).
 * Sirf nativeStop() kisi bhi thread se call ho sakta hai.
 */
public final class LlamaBridge {

    static {
        System.loadLibrary("llama_jni");
    }

    public static final int FINISH_STOP = 0;
    public static final int FINISH_LENGTH = 1;
    public static final int FINISH_CANCELLED = 2;

    /** Native code har token ka UTF-8 hissa yahan bhejta hai. Exception throw karo to generation ruk jayegi. */
    public interface TokenCallback {
        void onToken(byte[] utf8);
    }

    private LlamaBridge() {}

    /**
     * @param templateOverride "" = auto detect, "gemma4", ya llama.cpp template naam ("chatml", "llama3"...)
     * @return null agar success, warna error message
     */
    public static native String nativeLoad(String modelPath, int nCtx, int nThreads, String templateOverride);

    public static native boolean nativeIsLoaded();

    /**
     * OpenAI jaisa completion. roles/contents = poori chat. Blocking call.
     * @return reply text, error pe "[ERROR] ..." se shuru
     */
    public static native String nativeComplete(String[] roles, String[] contents,
                                               int maxTokens, float temperature, float topP, int topK,
                                               float frequencyPenalty, float presencePenalty,
                                               boolean thinking, TokenCallback callback);

    /** Last request: [promptTokens, completionTokens, reusedTokens, finishReason] */
    public static native int[] nativeLastStats();

    public static native void nativeStop();

    public static native void nativeUnload();

    public static native String nativeSystemInfo();
}
