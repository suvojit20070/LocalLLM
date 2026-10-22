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

import java.io.File;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.Collections;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Background service:
 *  - model ko RAM me loaded rakhta hai
 *  - local OpenAI-compatible server chalata hai (http://127.0.0.1:PORT/v1)
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
     * Isse load / unload / generate hamesha line se chalte hain, aur purane service
     * object ka unload naye object ke load ke baad nahi chal sakta.
     */
    private static final ExecutorService WORKER = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "llm-worker");
        t.setPriority(Thread.NORM_PRIORITY + 1);
        return t;
    });

    public enum State { STOPPED, NO_MODEL, LOADING, READY, GENERATING, ERROR }

    public interface StatusListener {
        void onStatusChanged(State state, String message);
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
    private PowerManager.WakeLock wakeLock;
    private OpenAiServer server;

    private volatile boolean alive = false;
    private volatile State state = State.STOPPED;
    private volatile String statusMessage = "Service band hai";
    private volatile String loadedPath = null;
    private volatile String modelId = "local-model";

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

    /** Settings badli -> model + server dobara */
    public static void reload(Context ctx) {
        ContextCompat.startForegroundService(ctx,
                new Intent(ctx, LlmService.class).setAction(ACTION_RELOAD));
    }

    // ------------------------------------------------------------------ lifecycle

    @Override
    public void onCreate() {
        super.onCreate();
        prefs = new Prefs(this);
        createChannel();
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "LocalLLM:generate");
        wakeLock.setReferenceCounted(false);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent != null ? intent.getAction() : null;

        // startForegroundService ke baad 5 sec ke andar ye zaroori hai
        goForeground(statusMessage);

        if (ACTION_STOP.equals(action)) {
            shutdown();
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE);
            stopSelf();
            return START_NOT_STICKY;
        }

        // System ne restart kiya (intent null) lekin user ne Stop kiya tha
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
        } else if (state != State.LOADING && (reload || !isReady()
                || !prefs.getModelPath().equals(loadedPath))) {
            loadModel(prefs.getModelPath());
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
        if (server != null) {
            server.stop();
            server = null;
        }
        LlamaBridge.nativeStop();
        WORKER.execute(LlamaBridge::nativeUnload);   // line me lagta hai, baad wale load se pehle chalega
        loadedPath = null;
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

    /** LAN access on ho to WiFi IP wala URL, warna null */
    @Nullable
    public String getLanEndpoint() {
        if (!prefs.isLanAccess()) return null;
        String ip = findLanIp();
        return ip == null ? null : "http://" + ip + ":" + prefs.getPort() + "/v1";
    }

    public boolean isServerRunning() {
        return server != null && server.isRunning();
    }

    public int getServedRequests() { return servedRequests.get(); }

    // ------------------------------------------------------------------ model

    public State getState() { return state; }

    public void addListener(StatusListener l) {
        listeners.addIfAbsent(l);
        main.post(() -> l.onStatusChanged(state, statusMessage));
    }

    public void removeListener(StatusListener l) {
        listeners.remove(l);
    }

    private void loadModel(String path) {
        setState(State.LOADING, "Model load ho raha hai…");
        loadedPath = null;
        WORKER.execute(() -> {
            if (!alive) return;   // beech me Stop ho gaya
            if (!new File(path).isFile()) {
                setState(State.NO_MODEL, "Model file nahi mili");
                return;
            }
            long t0 = SystemClock.elapsedRealtime();
            String err = LlamaBridge.nativeLoad(path, prefs.getContextSize(), prefs.getThreads(),
                    prefs.getTemplateOverride());
            long secs = (SystemClock.elapsedRealtime() - t0) / 1000;
            if (err == null) {
                loadedPath = path;
                modelId = prefs.getModelId();
                Log.i(TAG, LlamaBridge.nativeSystemInfo());
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

    // ------------------------------------------------------------------ OpenAiServer.Engine

    /** Java aur native dono ke hisaab se model ready ho */
    @Override
    public boolean isReady() {
        return alive && loadedPath != null && LlamaBridge.nativeIsLoaded();
    }

    @Override
    public String getStatusMessage() { return statusMessage; }

    @Override
    public String getModelId() { return modelId; }

    @Override
    public String getSystemPrompt() { return prefs.getSystemPrompt(); }

    @Override
    public int getDefaultMaxTokens() { return prefs.getDefaultMaxTokens(); }

    @Override
    public boolean isThinkingDefault() { return prefs.isThinking(); }

    /** Blocking. Server ke client threads se call hota hai; requests worker pe line me lagti hain. */
    @Override
    public OpenAiServer.Completion complete(String[] roles, String[] contents,
                                            OpenAiServer.GenParams p, LlamaBridge.TokenCallback cb) {
        if (!isReady()) {
            // Java ko laga loaded hai par native me model nahi -> khud reload karo
            if (alive && loadedPath != null && state != State.LOADING) {
                Log.w(TAG, "native model missing, reload kar rahe hain");
                main.post(() -> loadModel(prefs.getModelPath()));
                return OpenAiServer.Completion.error("Model reload ho raha hai, 30 sec baad dobara try karo");
            }
            return OpenAiServer.Completion.error("Model ready nahi hai: " + statusMessage);
        }

        if (activeRequests.incrementAndGet() == 1) setState(State.GENERATING, "Reply bana raha hai…");
        try {
            Future<OpenAiServer.Completion> f = WORKER.submit(() -> {
                wakeLock.acquire(10 * 60 * 1000L); // screen off me bhi CPU chalta rahe
                try {
                    OpenAiServer.Completion c = new OpenAiServer.Completion();
                    String text = LlamaBridge.nativeComplete(roles, contents, p.maxTokens,
                            p.temperature, p.topP, p.topK, p.frequencyPenalty, p.presencePenalty,
                            p.thinking, cb);
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
                    return c;
                } finally {
                    if (wakeLock.isHeld()) wakeLock.release();
                }
            });
            return f.get();
        } catch (Exception e) {
            Log.e(TAG, "complete fail", e);
            return OpenAiServer.Completion.error(String.valueOf(e.getMessage()));
        } finally {
            servedRequests.incrementAndGet();
            if (activeRequests.decrementAndGet() == 0 && isReady()) {
                setState(State.READY, "Ready • " + servedRequests.get() + " requests serve hui");
            }
        }
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
