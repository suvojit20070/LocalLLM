package com.suvo.localllm;

/**
 * llama.cpp ka Java wrapper (JNI).
 * load / complete / bench / unload sab ek hi background thread se call karo (LlmService.WORKER).
 * Sirf nativeStop() aur nativeIsLoaded() kisi bhi thread se call ho sakte hain.
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
     * @param nCtx         0 = model ka max (8192 tak)
     * @param flashAttn    -1 auto, 0 off, 1 on
     * @param kvType       0 f16, 1 q8_0, 2 q4_0
     * @param genCpus      generation threads kin cores pe (null/khaali = koi bhi)
     * @param batchCpus    prefill threads kin cores pe (null/khaali = koi bhi)
     * @return null agar success, warna error message
     */
    public static native String nativeLoad(String modelPath, int nCtx,
                                           int nThreads, int nThreadsBatch,
                                           int nBatch, int nUbatch,
                                           int flashAttn, int kvType, boolean useMmap,
                                           int[] genCpus, int[] batchCpus,
                                           String templateOverride);

    public static native boolean nativeIsLoaded();

    /** Model aur current settings ki jaankari (JSON) */
    public static native String nativeModelInfo();

    /**
     * OpenAI jaisa completion. roles/contents = poori chat. Blocking call.
     * @param seed -1 = random
     * @return reply text, error pe "[ERROR] ..." se shuru
     */
    public static native String nativeComplete(String[] roles, String[] contents,
                                               int maxTokens, float temperature, float topP, int topK,
                                               float minP, float repeatPenalty, int repeatLastN,
                                               float frequencyPenalty, float presencePenalty, int seed,
                                               boolean thinking, TokenCallback callback);

    /** Last request: [promptTokens, completionTokens, reusedTokens, finishReason, prefillMs, genMs] */
    public static native int[] nativeLastStats();

    /** Speed test: [prefill tok/s, generate tok/s] */
    public static native float[] nativeBench(int promptTokens, int genTokens);

    public static native void nativeStop();

    public static native void nativeUnload();

    public static native String nativeSystemInfo();
}
