package com.suvo.localllm;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Binder;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;
import androidx.core.app.ServiceCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.File;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Background service:
 *  - model ko RAM me loaded rakhta hai
 *  - local OpenAI-compatible server chalata hai (http://127.0.0.1:PORT/v1)
 *  - self-learning: khaali time me chats se user ke baare me baatein seekh ke memory me rakhta hai
 *  - system kill kare to START_STICKY se wapas aata hai, boot pe BootReceiver start karta hai
 */
public class LlmService extends Service implements OpenAiServer.Engine {

    private static final String TAG = "LlmService";

    public static final String ACTION_START  = "com.suvo.localllm.action.START";
    public static final String ACTION_STOP   = "com.suvo.localllm.action.STOP";
    public static final String ACTION_RELOAD = "com.suvo.localllm.action.RELOAD";

    private static final String CHANNEL_ID = "llm_service";
    private static final int NOTIF_ID = 101;

    /**
     * Poore process ka EK hi native worker thread (kabhi band nahi hota).
     * load / unload / generate / learn / bench hamesha line se chalte hain.
     */
    private static final ExecutorService WORKER = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "llm-worker");
        t.setPriority(Thread.MAX_PRIORITY);
        return t;
    });

    public enum State { STOPPED, NO_MODEL, LOADING, READY, GENERATING, LEARNING, BENCHMARK, ERROR }

    public interface StatusListener {
        void onStatusChanged(State state, String message);
    }

    public interface BenchCallback {
        void onResult(float prefillTps, float genTps, String error);
    }

    public class LocalBinder extends Binder {
        public LlmService getService() { return LlmService.this; }
    }

    // ------------------------------------------------------------------

    private final IBinder binder = new LocalBinder();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final CopyOnWriteArrayList<StatusListener> listeners = new CopyOnWriteArrayList<>();
    private final AtomicInteger activeRequests = new AtomicInteger(0);
    private final AtomicInteger servedRequests = new AtomicInteger(0);

    private Prefs prefs;
    private MemoryStore memory;
    private PowerManager.WakeLock wakeLock;
    private OpenAiServer server;

    private volatile boolean alive = false;
    private volatile State state = State.STOPPED;
    private volatile String statusMessage = "Service band hai";
    private volatile String loadedKey = null;      // kis setting se model load hua
    private volatile String modelId = "local-model";
    private volatile String lastTiming = "";
    private volatile String lastBench = "";
    private volatile String modelInfoCache = "{}";   // native lock ke bina padhne ke liye

    // self-learning
    private final ArrayDeque<String[]> pendingExchanges = new ArrayDeque<>();
    private volatile boolean learning = false;
    private volatile boolean learnAbort = false;
    private final Runnable learnCheck = this::maybeLearn;

    // ------------------------------------------------------------------ helpers for others

    public static void start(Context ctx) {
        new Prefs(ctx).setServiceEnabled(true);
        ContextCompat.startForegroundService(ctx,
                new Intent(ctx, LlmService.class).setAction(ACTION_START));
    }

    public static void stop(Context ctx) {
        new Prefs(ctx).setServiceEnabled(false);
        ctx.startService(new Intent(ctx, LlmService.class).setAction(ACTION_STOP));
    }

    /** Settings badli -> server restart, aur load settings badli ho to model reload */
    public static void reload(Context ctx) {
        ContextCompat.startForegroundService(ctx,
                new Intent(ctx, LlmService.class).setAction(ACTION_RELOAD));
    }

    // ------------------------------------------------------------------ lifecycle

    @Override
    public void onCreate() {
        super.onCreate();
        prefs = new Prefs(this);
        memory = new MemoryStore(this);
        createChannel();
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "LocalLLM:generate");
        wakeLock.setReferenceCounted(false);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent != null ? intent.getAction() : null;

        goForeground(statusMessage);   // startForegroundService ke 5 sec ke andar zaroori

        if (ACTION_STOP.equals(action)) {
            shutdown();
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE);
            stopSelf();
            return START_NOT_STICKY;
        }

        if (action == null && !prefs.isServiceEnabled()) {
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE);
            stopSelf();
            return START_NOT_STICKY;
        }

        alive = true;
        boolean reload = ACTION_RELOAD.equals(action);
        startServer(reload);

        if (!prefs.hasModel()) {
            setState(State.NO_MODEL, "Model nahi hai. App kholke download/select karo");
        } else if (state != State.LOADING && (!isReady() || !prefs.loadKey().equals(loadedKey))) {
            loadModel();
        }
        return START_STICKY;
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    @Override
    public void onDestroy() {
        if (alive) shutdown();
        super.onDestroy();
    }

    private void shutdown() {
        alive = false;
        main.removeCallbacks(learnCheck);
        if (server != null) {
            server.stop();
            server = null;
        }
        LlamaBridge.nativeStop();
        WORKER.execute(LlamaBridge::nativeUnload);
        loadedKey = null;
        state = State.STOPPED;
        statusMessage = "Service band hai";
        notifyListeners();
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
    }

    // ------------------------------------------------------------------ server

    private void startServer(boolean restart) {
        if (server != null && server.isRunning() && !restart) return;
        if (server != null) server.stop();
        server = new OpenAiServer(this, prefs.getPort(), prefs.isLanAccess(), prefs.getApiKey());
        try {
            server.start();
        } catch (Exception e) {
            Log.e(TAG, "server start fail", e);
            server = null;
            setState(State.ERROR, "Port " + prefs.getPort() + " busy hai, Settings me port badlo");
        }
    }

    public String getEndpoint() {
        return "http://127.0.0.1:" + prefs.getPort() + "/v1";
    }

    @Nullable
    public String getLanEndpoint() {
        if (!prefs.isLanAccess()) return null;
        String ip = findLanIp();
        return ip == null ? null : "http://" + ip + ":" + prefs.getPort() + "/v1";
    }

    public boolean isServerRunning() {
        return server != null && server.isRunning();
    }

    // ------------------------------------------------------------------ model

    public State getState() { return state; }

    public String getLastTiming() { return lastTiming; }

    public String getLastBench() { return lastBench; }

    public void addListener(StatusListener l) {
        listeners.addIfAbsent(l);
        main.post(() -> l.onStatusChanged(state, statusMessage));
    }

    public void removeListener(StatusListener l) {
        listeners.remove(l);
    }

    private void loadModel() {
        final String key = prefs.loadKey();
        final String path = prefs.getModelPath();
        setState(State.LOADING, "Model load ho raha hai…");
        loadedKey = null;
        WORKER.execute(() -> {
            if (!alive) return;
            if (!new File(path).isFile()) {
                setState(State.NO_MODEL, "Model file nahi mili");
                return;
            }
            CpuInfo cpu = CpuInfo.get();
            int[] genCpus = prefs.isFastCoresGen() ? cpu.fastCores : new int[0];
            int[] batchCpus = prefs.isFastCoresBatch() ? cpu.fastCores : new int[0];

            long t0 = SystemClock.elapsedRealtime();
            String err = LlamaBridge.nativeLoad(path, prefs.getContextSize(),
                    prefs.getThreads(), prefs.getThreadsBatch(),
                    prefs.getBatch(), prefs.getUbatch(),
                    prefs.getFlashAttn(), prefs.getKvType(), prefs.isMmap(),
                    genCpus, batchCpus, prefs.getTemplateOverride());
            long secs = (SystemClock.elapsedRealtime() - t0) / 1000;
            if (err == null) {
                loadedKey = key;
                modelId = prefs.getModelId();
                modelInfoCache = LlamaBridge.nativeModelInfo();
                Log.i(TAG, modelInfoCache);
                if (isServerRunning()) {
                    setState(State.READY, "Ready (" + secs + "s me load hua)");
                } else {
                    setState(State.ERROR, "Model ready, lekin port " + prefs.getPort() + " busy hai. Port badlo");
                }
            } else {
                Log.e(TAG, "load fail: " + err);
                setState(State.ERROR, err);
            }
        });
    }

    /** Speed test (prefill + generate tok/s). Callback main thread pe. */
    public void runBenchmark(BenchCallback cb) {
        if (!isReady()) {
            main.post(() -> cb.onResult(-1, -1, "Model ready nahi hai"));
            return;
        }
        if (activeRequests.get() > 0) {
            main.post(() -> cb.onResult(-1, -1, "Abhi request chal rahi hai, baad me try karo"));
            return;
        }
        if (learning) LlamaBridge.nativeStop();
        setState(State.BENCHMARK, "Speed test chal raha hai…");
        WORKER.execute(() -> {
            wakeLock.acquire(5 * 60 * 1000L);
            float[] r;
            try {
                r = LlamaBridge.nativeBench(128, 32);
            } finally {
                if (wakeLock.isHeld()) wakeLock.release();
            }
            final float pp = r[0], tg = r[1];
            if (pp > 0) {
                lastBench = String.format(Locale.US, "Prefill %.1f tok/s • Generate %.1f tok/s", pp, tg);
            }
            setState(State.READY, pp > 0 ? "Ready • " + lastBench : "Ready");
            main.post(() -> cb.onResult(pp, tg, pp > 0 ? null : "Benchmark fail"));
        });
    }

    // ------------------------------------------------------------------ OpenAiServer.Engine

    /** Java aur native dono ke hisaab se model ready ho */
    @Override
    public boolean isReady() {
        return alive && loadedKey != null && LlamaBridge.nativeIsLoaded();
    }

    @Override
    public String getStatusMessage() { return statusMessage; }

    @Override
    public String getModelId() { return modelId; }

    @Override
    public String buildSystemPrompt(String requestSystem, boolean useMemory) {
        String base = requestSystem != null ? requestSystem : prefs.getSystemPrompt();
        if (base == null) base = "";
        if (useMemory && prefs.isUseMemory()) {
            String block = memory.promptBlock();
            if (!block.isEmpty()) base = base.isEmpty() ? block.trim() : base + block;
        }
        return base;
    }

    @Override
    public OpenAiServer.GenParams defaultParams() {
        OpenAiServer.GenParams p = new OpenAiServer.GenParams();
        p.maxTokens = prefs.getDefaultMaxTokens();
        p.temperature = prefs.getTemperature();
        p.topP = prefs.getTopP();
        p.topK = prefs.getTopK();
        p.minP = prefs.getMinP();
        p.repeatPenalty = prefs.getRepeatPenalty();
        p.repeatLastN = prefs.getRepeatLastN();
        p.thinking = prefs.isThinking();
        p.learn = prefs.isLearning();
        p.useMemory = prefs.isUseMemory();
        return p;
    }

    @Override
    public MemoryStore memory() { return memory; }

    @Override
    public int maxMemories() { return prefs.getMaxMemories(); }

    @Override
    public String infoJson() {
        try {
            // cache se: generation ke time native lock pe UI atak jata
            JSONObject o = new JSONObject(isReady() ? modelInfoCache : "{}");
            o.put("model_id", modelId);
            o.put("ready", isReady());
            o.put("status", statusMessage);
            o.put("cpu", CpuInfo.get().describe());
            o.put("served_requests", servedRequests.get());
            o.put("memories", memory.size());
            o.put("last_timing", lastTiming);
            o.put("last_bench", lastBench);
            return o.toString();
        } catch (Exception e) {
            return "{}";
        }
    }

    /** Blocking. Server ke client threads se call hota hai; requests worker pe line me lagti hain. */
    @Override
    public OpenAiServer.Completion complete(String[] roles, String[] contents,
                                            OpenAiServer.GenParams p, LlamaBridge.TokenCallback cb) {
        if (!isReady()) {
            if (alive && loadedKey != null && state != State.LOADING) {
                Log.w(TAG, "native model missing, reload kar rahe hain");
                main.post(this::loadModel);
                return OpenAiServer.Completion.error("Model reload ho raha hai, 30 sec baad dobara try karo");
            }
            return OpenAiServer.Completion.error("Model ready nahi hai: " + statusMessage);
        }

        // User ki request sabse pehle: background learning chal rahi ho to rok do
        main.removeCallbacks(learnCheck);
        learnAbort = true;
        if (learning) LlamaBridge.nativeStop();

        if (activeRequests.incrementAndGet() == 1) setState(State.GENERATING, "Reply bana raha hai…");
        try {
            Future<OpenAiServer.Completion> f = WORKER.submit(() -> {
                wakeLock.acquire(10 * 60 * 1000L);
                try {
                    return runNative(roles, contents, p, cb);
                } finally {
                    if (wakeLock.isHeld()) wakeLock.release();
                }
            });
            OpenAiServer.Completion c = f.get();
            if (c.error == null) {
                double pps = c.prefillMs > 0 ? (c.promptTokens - c.reusedTokens) * 1000.0 / c.prefillMs : 0;
                double gps = c.genMs > 0 ? c.completionTokens * 1000.0 / c.genMs : 0;
                lastTiming = String.format(Locale.US,
                        "prompt %d (cache %d) %.1fs @ %.0f t/s • reply %d @ %.1f t/s",
                        c.promptTokens, c.reusedTokens, c.prefillMs / 1000.0, pps, c.completionTokens, gps);
            }
            return c;
        } catch (Exception e) {
            Log.e(TAG, "complete fail", e);
            return OpenAiServer.Completion.error(String.valueOf(e.getMessage()));
        } finally {
            servedRequests.incrementAndGet();
            if (activeRequests.decrementAndGet() == 0 && isReady()) {
                setState(State.READY, "Ready • " + lastTiming);
                scheduleLearn();
            }
        }
    }

    /** WORKER thread pe hi call karo */
    private OpenAiServer.Completion runNative(String[] roles, String[] contents,
                                              OpenAiServer.GenParams p, LlamaBridge.TokenCallback cb) {
        OpenAiServer.Completion c = new OpenAiServer.Completion();
        String text = LlamaBridge.nativeComplete(roles, contents, p.maxTokens,
                p.temperature, p.topP, p.topK, p.minP, p.repeatPenalty, p.repeatLastN,
                p.frequencyPenalty, p.presencePenalty, p.seed, p.thinking, cb);
        int[] st = LlamaBridge.nativeLastStats();
        if (text != null && text.startsWith("[ERROR]")) {
            c.error = text.substring(7).trim();
        } else {
            c.text = text == null ? "" : text;
        }
        c.promptTokens = st[0];
        c.completionTokens = st[1];
        c.reusedTokens = st[2];
        c.finish = st[3];
        c.prefillMs = st[4];
        c.genMs = st[5];
        return c;
    }

    // ------------------------------------------------------------------ self learning

    @Override
    public void onExchange(String userText, String reply) {
        if (!prefs.isLearning() || userText == null || userText.trim().isEmpty()) return;
        synchronized (pendingExchanges) {
            pendingExchanges.addLast(new String[]{userText, reply == null ? "" : reply});
            while (pendingExchanges.size() > 6) pendingExchanges.removeFirst();
        }
        scheduleLearn();
    }

    private void scheduleLearn() {
        main.removeCallbacks(learnCheck);
        boolean has;
        synchronized (pendingExchanges) { has = !pendingExchanges.isEmpty(); }
        if (has && prefs.isLearning()) {
            main.postDelayed(learnCheck, prefs.getLearnIdleSeconds() * 1000L);
        }
    }

    /** Phone khaali hai to pichli chats se user ke baare me nayi baatein nikaalo */
    private void maybeLearn() {
        if (!alive || !isReady() || activeRequests.get() > 0 || learning) {
            scheduleLearn();
            return;
        }
        final List<String[]> batch;
        synchronized (pendingExchanges) {
            if (pendingExchanges.isEmpty()) return;
            batch = new ArrayList<>(pendingExchanges);
            pendingExchanges.clear();
        }

        StringBuilder chat = new StringBuilder();
        for (String[] ex : batch) {
            chat.append("USER: ").append(shorten(ex[0], 600)).append('\n');
            chat.append("AI: ").append(shorten(stripThought(ex[1]), 400)).append("\n\n");
        }
        String system = "Tum ek memory note-taker ho. Neeche ki chat padho aur USER ke baare me sirf NAYI, "
                + "kaam ki, lambe samay tak sach rehne wali baatein likho: naam, umar, shehar, kaam/padhai, "
                + "pasand-naapasand, plans, aur wo kaise jawab chahta hai. AI ki baatein ya ek baar ke sawaal mat likho. "
                + "Har baat ek chhoti line me '- ' se shuru karo, user ki bhasha me. "
                + "Kuch naya na ho to sirf NONE likho.";
        String user = "Pehle se pata hai:\n" + memory.bulletList() + "\n\nChat:\n" + chat.toString().trim();

        learning = true;
        learnAbort = false;
        setState(State.LEARNING, "Chat se seekh raha hai…");
        WORKER.execute(() -> {
            int added = 0;
            boolean cancelled = false;
            try {
                if (learnAbort) {           // shuru hone se pehle hi user ki request aa gayi
                    cancelled = true;
                } else {
                    OpenAiServer.GenParams p = new OpenAiServer.GenParams();
                    p.maxTokens = 160;
                    p.temperature = 0.2f;
                    p.topK = 20;
                    p.thinking = false;
                    OpenAiServer.Completion c = runNative(new String[]{"system", "user"},
                            new String[]{system, user}, p, null);
                    cancelled = c.finish == LlamaBridge.FINISH_CANCELLED || learnAbort;
                    if (c.error == null && !cancelled) {
                        added = memory.addFromModelOutput(c.text, prefs.getMaxMemories());
                        Log.i(TAG, "learned " + added + ": " + c.text);
                    }
                }
            } catch (Throwable t) {
                Log.w(TAG, "learn fail", t);
            } finally {
                learning = false;
            }
            if (cancelled) {
                // user ki request aa gayi thi; baad me dobara try
                synchronized (pendingExchanges) {
                    for (int i = batch.size() - 1; i >= 0; i--) pendingExchanges.addFirst(batch.get(i));
                }
            }
            if (activeRequests.get() == 0 && isReady()) {
                setState(State.READY, added > 0
                        ? "Ready • " + added + " nayi baat seekhi (" + memory.size() + " yaad)"
                        : "Ready • " + lastTiming);
            }
            if (cancelled) main.post(this::scheduleLearn);
        });
    }

    private static String shorten(String s, int max) {
        String t = s == null ? "" : s.trim();
        return t.length() > max ? t.substring(0, max) + "…" : t;
    }

    private static String stripThought(String s) {
        if (s == null) return "";
        int a = s.indexOf("<thought>");
        int b = s.indexOf("</thought>");
        if (a >= 0 && b > a) return (s.substring(0, a) + s.substring(b + 10)).trim();
        return s;
    }

    // ------------------------------------------------------------------ internals

    private void setState(State s, String msg) {
        state = s;
        statusMessage = msg;
        updateNotification();
        notifyListeners();
    }

    private void notifyListeners() {
        State s = state;
        String m = statusMessage;
        main.post(() -> {
            for (StatusListener l : listeners) l.onStatusChanged(s, m);
        });
    }

    @Nullable
    private static String findLanIp() {
        try {
            for (NetworkInterface ni : Collections.list(NetworkInterface.getNetworkInterfaces())) {
                if (!ni.isUp() || ni.isLoopback()) continue;
                String name = ni.getName();
                if (!name.startsWith("wlan") && !name.startsWith("ap") && !name.startsWith("swlan")) continue;
                for (InetAddress a : Collections.list(ni.getInetAddresses())) {
                    if (a instanceof Inet4Address && !a.isLoopbackAddress()) return a.getHostAddress();
                }
            }
        } catch (Exception ignored) { }
        return null;
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel ch = new NotificationChannel(
                    CHANNEL_ID, "Local LLM server", NotificationManager.IMPORTANCE_LOW);
            ch.setDescription("Model background me loaded hai");
            ch.setShowBadge(false);
            getSystemService(NotificationManager.class).createNotificationChannel(ch);
        }
    }

    private Notification buildNotification(String text) {
        Intent open = new Intent(this, MainActivity.class)
                .setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent openPi = PendingIntent.getActivity(this, 0, open,
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);

        Intent stop = new Intent(this, LlmService.class).setAction(ACTION_STOP);
        PendingIntent stopPi = PendingIntent.getService(this, 1, stop,
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);

        String endpoint = (prefs != null) ? getEndpoint() : "";
        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle("Local LLM • " + endpoint)
                .setContentText(text)
                .setStyle(new NotificationCompat.BigTextStyle().bigText(text))
                .setContentIntent(openPi)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setSilent(true)
                .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
                .addAction(android.R.drawable.ic_menu_close_clear_cancel, "Stop", stopPi)
                .build();
    }

    private void goForeground(String text) {
        int type = Build.VERSION.SDK_INT >= 34 ? ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE : 0;
        ServiceCompat.startForeground(this, NOTIF_ID, buildNotification(text), type);
    }

    private void updateNotification() {
        try {
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null && state != State.STOPPED) nm.notify(NOTIF_ID, buildNotification(statusMessage));
        } catch (SecurityException ignored) {
            // notification permission nahi hai, service phir bhi chalega
        }
    }
}
