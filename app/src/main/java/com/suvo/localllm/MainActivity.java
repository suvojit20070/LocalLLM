package com.suvo.localllm;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.OpenableColumns;
import android.provider.Settings;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.materialswitch.MaterialSwitch;
import com.google.android.material.progressindicator.CircularProgressIndicator;
import com.google.android.material.progressindicator.LinearProgressIndicator;
import com.google.android.material.textfield.TextInputEditText;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;

/** Sirf settings screen: model, system prompt, server settings. Chat koi nahi, endpoint use karo. */
public class MainActivity extends AppCompatActivity implements LlmService.StatusListener {

    private TextView tvStatus, tvEndpoint, tvModelId, tvModel, tvDownload, tvBattery;
    private CircularProgressIndicator progressStatus;
    private LinearProgressIndicator progressDownload;
    private LinearLayout downloadBox;
    private MaterialButton btnService, btnCopy, btnDownload, btnPick, btnSavePrompt, btnApply, btnBattery;
    private TextInputEditText etSystem, etPort, etCtx, etThreads, etMaxTokens, etApiKey;
    private MaterialSwitch swLan, swAutoStart, swThinking;

    private Prefs prefs;
    private LlmService service;
    private boolean bound;
    private boolean copying;
    private LlmService.State lastState = LlmService.State.STOPPED;

    private final Handler ui = new Handler(Looper.getMainLooper());
    private final Runnable downloadPoll = new Runnable() {
        @Override
        public void run() {
            if (pollDownload()) ui.postDelayed(this, 800);
        }
    };

    // ------------------------------------------------------------------ launchers

    private final ActivityResultLauncher<String[]> pickModel =
            registerForActivityResult(new ActivityResultContracts.OpenDocument(), uri -> {
                if (uri != null) copyModelToApp(uri);
            });

    private final ActivityResultLauncher<String> notifPermission =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> { });

    private final ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder binder) {
            service = ((LlmService.LocalBinder) binder).getService();
            service.addListener(MainActivity.this);
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            service = null;
            onStatusChanged(LlmService.State.STOPPED, "Service band hai");
        }
    };

    // ------------------------------------------------------------------ lifecycle

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        prefs = new Prefs(this);

        tvStatus = findViewById(R.id.tvStatus);
        tvEndpoint = findViewById(R.id.tvEndpoint);
        tvModelId = findViewById(R.id.tvModelId);
        tvModel = findViewById(R.id.tvModel);
        tvDownload = findViewById(R.id.tvDownload);
        tvBattery = findViewById(R.id.tvBattery);
        progressStatus = findViewById(R.id.progressStatus);
        progressDownload = findViewById(R.id.progressDownload);
        downloadBox = findViewById(R.id.downloadBox);
        btnService = findViewById(R.id.btnService);
        btnCopy = findViewById(R.id.btnCopy);
        btnDownload = findViewById(R.id.btnDownload);
        btnPick = findViewById(R.id.btnPick);
        btnSavePrompt = findViewById(R.id.btnSavePrompt);
        btnApply = findViewById(R.id.btnApply);
        btnBattery = findViewById(R.id.btnBattery);
        etSystem = findViewById(R.id.etSystem);
        etPort = findViewById(R.id.etPort);
        etCtx = findViewById(R.id.etCtx);
        etThreads = findViewById(R.id.etThreads);
        etMaxTokens = findViewById(R.id.etMaxTokens);
        etApiKey = findViewById(R.id.etApiKey);
        swLan = findViewById(R.id.swLan);
        swAutoStart = findViewById(R.id.swAutoStart);
        swThinking = findViewById(R.id.swThinking);

        loadSettingsIntoViews();

        btnService.setOnClickListener(v -> {
            if (isServiceActive()) {
                LlmService.stop(this);
            } else if (!prefs.hasModel()) {
                toast("Pehle model download karo ya file chuno");
            } else {
                startService();
            }
        });

        btnCopy.setOnClickListener(v -> {
            ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            cm.setPrimaryClip(ClipData.newPlainText("endpoint", baseUrl()));
            toast("Copy ho gaya");
        });

        btnDownload.setOnClickListener(v -> {
            if (prefs.getDownloadId() != -1) {
                ModelDownloader.cancel(this);
                refreshModelUi();
            } else {
                confirmDownload();
            }
        });

        btnPick.setOnClickListener(v -> {
            if (!copying) pickModel.launch(new String[]{"*/*"});
        });

        btnSavePrompt.setOnClickListener(v -> {
            prefs.setSystemPrompt(textOf(etSystem));
            toast("System prompt save ho gaya (agli request se lagega)");
        });

        btnApply.setOnClickListener(v -> applySettings());

        swAutoStart.setOnCheckedChangeListener((b, checked) -> prefs.setAutoStart(checked));
        // per-request setting hai, model reload nahi chahiye
        swThinking.setOnCheckedChangeListener((b, checked) -> prefs.setThinking(checked));

        btnBattery.setOnClickListener(v -> askIgnoreBatteryOptimization());

        askNotificationPermission();
    }

    @Override
    protected void onStart() {
        super.onStart();
        bindToService();
    }

    @Override
    protected void onResume() {
        super.onResume();
        ModelDownloader.finishIfDone(this);
        refreshModelUi();
        refreshBatteryUi();
        if (prefs.getDownloadId() != -1) {
            ui.removeCallbacks(downloadPoll);
            ui.post(downloadPoll);
        }
        // Model hai aur user ne stop nahi kiya -> server chalu rakho
        if (prefs.hasModel() && prefs.isServiceEnabled() && !isServiceActive()) {
            startService();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        ui.removeCallbacks(downloadPoll);
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (service != null) service.removeListener(this);
        if (bound) {
            unbindService(connection);
            bound = false;
        }
        service = null;
    }

    private void bindToService() {
        if (bound) return;
        // flag 0: bind karne se service start nahi hota, chal raha ho tab connect hota hai
        bound = bindService(new Intent(this, LlmService.class), connection, 0);
    }

    private void startService() {
        LlmService.start(this);
        bindToService();
    }

    // ------------------------------------------------------------------ status

    @Override
    public void onStatusChanged(LlmService.State state, String message) {
        lastState = state;
        tvStatus.setText(message);
        boolean busy = state == LlmService.State.LOADING || state == LlmService.State.GENERATING;
        progressStatus.setVisibility(busy ? View.VISIBLE : View.GONE);
        btnService.setText(isServiceActive() ? "Stop server" : "Start server");
        refreshEndpoint();
    }

    private boolean isServiceActive() {
        return service != null && lastState != LlmService.State.STOPPED;
    }

    private String baseUrl() {
        return "http://127.0.0.1:" + prefs.getPort() + "/v1";
    }

    private void refreshEndpoint() {
        StringBuilder sb = new StringBuilder(baseUrl());
        if (service != null) {
            String lanUrl = service.getLanEndpoint();
            if (lanUrl != null) sb.append("\n").append(lanUrl).append("  (WiFi)");
        }
        tvEndpoint.setText(sb.toString());
        String key = prefs.getApiKey().isEmpty() ? "koi bhi" : "set hai";
        tvModelId.setText("model: " + prefs.getModelId() + "\napi key: " + key);
    }

    // ------------------------------------------------------------------ settings

    private void loadSettingsIntoViews() {
        etSystem.setText(prefs.getSystemPrompt());
        etPort.setText(String.valueOf(prefs.getPort()));
        etCtx.setText(String.valueOf(prefs.getContextSize()));
        etThreads.setText(String.valueOf(prefs.getThreads()));
        etMaxTokens.setText(String.valueOf(prefs.getDefaultMaxTokens()));
        etApiKey.setText(prefs.getApiKey());
        swLan.setChecked(prefs.isLanAccess());
        swAutoStart.setChecked(prefs.isAutoStart());
        swThinking.setChecked(prefs.isThinking());
        refreshEndpoint();
    }

    private void applySettings() {
        int port = parse(etPort, -1);
        int ctx = parse(etCtx, -1);
        int threads = parse(etThreads, -1);
        int maxTok = parse(etMaxTokens, -1);

        if (port < 1024 || port > 65535) { toast("Port 1024 se 65535 ke beech rakho"); return; }
        if (ctx < 512 || ctx > 32768) { toast("Context size 512 se 32768 ke beech rakho"); return; }
        if (threads < 1 || threads > 16) { toast("Threads 1 se 16 ke beech rakho"); return; }
        if (maxTok < 16 || maxTok > ctx) { toast("Max tokens 16 se context size tak"); return; }

        prefs.setPort(port);
        prefs.setContextSize(ctx);
        prefs.setThreads(threads);
        prefs.setDefaultMaxTokens(maxTok);
        prefs.setApiKey(textOf(etApiKey));
        prefs.setLanAccess(swLan.isChecked());
        prefs.setSystemPrompt(textOf(etSystem));
        refreshEndpoint();

        if (isServiceActive()) {
            LlmService.reload(this);
            toast("Settings save. Server aur model restart ho rahe hain");
        } else {
            toast("Settings save ho gayi");
        }
    }

    // ------------------------------------------------------------------ model: download

    private void confirmDownload() {
        long free = Prefs.modelDir(this).getUsableSpace();
        if (free < 3500L * 1024 * 1024) {
            new AlertDialog.Builder(this)
                    .setTitle("Jagah kam hai")
                    .setMessage("Download ke liye kam se kam 3.5 GB khaali jagah chahiye. Abhi "
                            + (free / (1024 * 1024)) + " MB hai.")
                    .setPositiveButton("OK", null)
                    .show();
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("Model download karein?")
                .setMessage("Gemma 4 E2B (Q4_K_M), lagbhag 3.1 GB. Mobile data pe bhi chalega.\n"
                        + "App band kar doge tab bhi download chalta rahega.")
                .setPositiveButton("Download", (d, w) -> {
                    ModelDownloader.start(this);
                    refreshModelUi();
                    ui.removeCallbacks(downloadPoll);
                    ui.post(downloadPoll);
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    /** @return true = polling jaari rakho */
    private boolean pollDownload() {
        if (prefs.getDownloadId() == -1) {
            refreshModelUi();
            return false;
        }
        ModelDownloader.Progress p = ModelDownloader.query(this);
        if (p.status == ModelDownloader.STATUS_RUNNING) {
            downloadBox.setVisibility(View.VISIBLE);
            if (p.total > 0) {
                progressDownload.setIndeterminate(false);
                progressDownload.setProgressCompat(p.percent(), true);
                tvDownload.setText(String.format(Locale.US, "%d%%  •  %d / %d MB  %s",
                        p.percent(), p.downloaded >> 20, p.total >> 20, p.message));
            } else {
                progressDownload.setIndeterminate(true);
                tvDownload.setText(p.message.isEmpty() ? "Shuru ho raha hai…" : p.message);
            }
            return true;
        }

        if (p.status == ModelDownloader.STATUS_DONE) {
            boolean ok = ModelDownloader.finishIfDone(this);
            refreshModelUi();
            if (ok) {
                deleteOtherModels(new File(prefs.getModelPath()));
                toast("Model download ho gaya, server start ho raha hai");
                startService();
            } else {
                toast("Download hua lekin file theek nahi lagi, dobara try karo");
            }
            return false;
        }

        // fail
        String msg = p.message;
        ModelDownloader.cancel(this);
        refreshModelUi();
        tvModel.setText(msg.isEmpty() ? "Download fail ho gaya, dobara try karo" : msg);
        return false;
    }

    // ------------------------------------------------------------------ model: pick file

    /** Chuni hui .gguf ko app ke model folder me copy karte hain (koi storage permission nahi chahiye). */
    private void copyModelToApp(Uri uri) {
        String name = queryName(uri);
        long size = querySize(uri);
        if (name == null || !name.toLowerCase(Locale.ROOT).endsWith(".gguf")) {
            toast("Ye .gguf file nahi hai");
            return;
        }
        File dir = Prefs.modelDir(this);
        File out = new File(dir, name);

        if (out.isFile() && size > 0 && out.length() == size) {
            onModelSelected(out);
            return;
        }
        if (size > 0 && dir.getUsableSpace() < size + 200L * 1024 * 1024) {
            toast("Jagah kam hai: " + (size >> 20) + " MB chahiye");
            return;
        }

        copying = true;
        btnPick.setEnabled(false);
        btnDownload.setEnabled(false);
        downloadBox.setVisibility(View.VISIBLE);
        progressDownload.setIndeterminate(size <= 0);
        tvDownload.setText("Copy ho raha hai…");

        new Thread(() -> {
            File tmp = new File(dir, name + ".part");
            String err = null;
            try (InputStream in = getContentResolver().openInputStream(uri);
                 OutputStream os = new FileOutputStream(tmp)) {
                if (in == null) throw new IllegalStateException("File khul nahi rahi");
                byte[] buf = new byte[1 << 20];
                long done = 0;
                int lastPct = -1, n;
                while ((n = in.read(buf)) > 0) {
                    os.write(buf, 0, n);
                    done += n;
                    if (size > 0) {
                        int pct = (int) (done * 100 / size);
                        if (pct != lastPct) {
                            lastPct = pct;
                            runOnUiThread(() -> {
                                progressDownload.setProgressCompat(pct, true);
                                tvDownload.setText("Copy ho raha hai… " + pct + "%");
                            });
                        }
                    }
                }
                ((FileOutputStream) os).getFD().sync();
            } catch (Exception e) {
                err = e.getMessage();
            }

            if (err == null) {
                //noinspection ResultOfMethodCallIgnored
                out.delete();
                if (!tmp.renameTo(out)) err = "Rename fail";
            }
            if (err != null) {
                //noinspection ResultOfMethodCallIgnored
                tmp.delete();
            }

            final String error = err;
            runOnUiThread(() -> {
                copying = false;
                btnPick.setEnabled(true);
                btnDownload.setEnabled(true);
                downloadBox.setVisibility(View.GONE);
                if (error == null) {
                    onModelSelected(out);
                } else {
                    refreshModelUi();
                    tvModel.setText("Copy fail: " + error);
                }
            });
        }, "model-copy").start();
    }

    private void onModelSelected(File model) {
        prefs.setModelPath(model.getAbsolutePath());
        deleteOtherModels(model);
        refreshModelUi();
        refreshEndpoint();
        toast("Model set ho gaya, server start ho raha hai");
        startService();   // naya path hai to service khud reload karega
    }

    /** Purani model files hatao taaki 3GB+ jagah bekar na jaye */
    private void deleteOtherModels(File keep) {
        if (prefs.getDownloadId() != -1) return;
        File[] files = Prefs.modelDir(this).listFiles();
        if (files == null) return;
        for (File f : files) {
            if (!f.equals(keep)) {
                //noinspection ResultOfMethodCallIgnored
                f.delete();
            }
        }
    }

    private void refreshModelUi() {
        boolean downloading = prefs.getDownloadId() != -1;
        if (prefs.hasModel()) {
            File f = new File(prefs.getModelPath());
            tvModel.setText(f.getName() + "  (" + (f.length() >> 20) + " MB)");
            btnDownload.setVisibility(downloading ? View.VISIBLE : View.GONE);
            btnPick.setText("Doosra model chuno");
        } else {
            tvModel.setText(downloading ? "Model download ho raha hai…"
                    : "Model nahi hai. 1 click me download karo ya phone se .gguf file chuno.");
            btnDownload.setVisibility(View.VISIBLE);
            btnPick.setText("Phone se .gguf file chuno");
        }
        btnDownload.setText(downloading ? "Download cancel karo" : "Download Gemma 4 E2B (3.1 GB)");
        if (!copying) downloadBox.setVisibility(downloading ? View.VISIBLE : View.GONE);
        btnService.setEnabled(prefs.hasModel() || isServiceActive());
    }

    private String queryName(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst()) return c.getString(0);
        } catch (Exception ignored) { }
        return uri.getLastPathSegment();
    }

    private long querySize(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, new String[]{OpenableColumns.SIZE}, null, null, null)) {
            if (c != null && c.moveToFirst() && !c.isNull(0)) return c.getLong(0);
        } catch (Exception ignored) { }
        return -1;
    }

    // ------------------------------------------------------------------ permissions

    private void askNotificationPermission() {
        if (Build.VERSION.SDK_INT >= 33
                && ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            notifPermission.launch(Manifest.permission.POST_NOTIFICATIONS);
        }
    }

    private void refreshBatteryUi() {
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        boolean ignoring = pm.isIgnoringBatteryOptimizations(getPackageName());
        tvBattery.setText(ignoring
                ? "✓ Battery optimization off hai. Service background me chalti rahegi."
                : "Battery optimization on hai, Android kuch der baad server band kar sakta hai.");
        btnBattery.setVisibility(ignoring ? View.GONE : View.VISIBLE);
    }

    @SuppressLint("BatteryLife")
    private void askIgnoreBatteryOptimization() {
        try {
            Intent i = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
            i.setData(Uri.parse("package:" + getPackageName()));
            startActivity(i);
        } catch (Exception e) {
            startActivity(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS));
        }
    }

    // ------------------------------------------------------------------ utils

    private static String textOf(TextInputEditText et) {
        CharSequence cs = et.getText();
        return cs == null ? "" : cs.toString().trim();
    }

    private static int parse(TextInputEditText et, int def) {
        try {
            return Integer.parseInt(textOf(et));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }
}
