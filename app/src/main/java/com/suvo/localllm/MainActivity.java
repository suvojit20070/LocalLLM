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
import com.google.android.material.button.MaterialButtonToggleGroup;
import com.google.android.material.materialswitch.MaterialSwitch;
import com.google.android.material.progressindicator.CircularProgressIndicator;
import com.google.android.material.progressindicator.LinearProgressIndicator;
import com.google.android.material.textfield.TextInputEditText;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.List;
import java.util.Locale;

/** Settings screen: model, presets, memory, performance, server. Chat ke liye endpoint use karo. */
public class MainActivity extends AppCompatActivity implements LlmService.StatusListener {

    private TextView tvStatus, tvEndpoint, tvModelId, tvTiming, tvModel, tvModelInfo, tvDownload,
            tvBattery, tvCpu, tvMemCount;
    private CircularProgressIndicator progressStatus;
    private LinearProgressIndicator progressDownload;
    private LinearLayout downloadBox;
    private MaterialButton btnService, btnBench, btnDownload, btnPick;
    private TextInputEditText etSystem, etMaxMem, etCtx, etBatch, etThreads, etThreadsBatch, etUbatch,
            etMaxTokens, etTemp, etTopK, etTopP, etMinP, etRepeat, etRepeatN, etPort, etApiKey;
    private MaterialSwitch swLearning, swUseMemory, swFastGen, swFastBatch, swMmap, swThinking, swLan, swAutoStart;
    private MaterialButtonToggleGroup tgFlash, tgKv;

    private Prefs prefs;
    private MemoryStore memoryView;   // sirf dikhane ke liye (service apna alag object rakhta hai)
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
        tvTiming = findViewById(R.id.tvTiming);
        tvModel = findViewById(R.id.tvModel);
        tvModelInfo = findViewById(R.id.tvModelInfo);
        tvDownload = findViewById(R.id.tvDownload);
        tvBattery = findViewById(R.id.tvBattery);
        tvCpu = findViewById(R.id.tvCpu);
        tvMemCount = findViewById(R.id.tvMemCount);
        progressStatus = findViewById(R.id.progressStatus);
        progressDownload = findViewById(R.id.progressDownload);
        downloadBox = findViewById(R.id.downloadBox);
        btnService = findViewById(R.id.btnService);
        btnBench = findViewById(R.id.btnBench);
        btnDownload = findViewById(R.id.btnDownload);
        btnPick = findViewById(R.id.btnPick);
        etSystem = findViewById(R.id.etSystem);
        etMaxMem = findViewById(R.id.etMaxMem);
        etCtx = findViewById(R.id.etCtx);
        etBatch = findViewById(R.id.etBatch);
        etThreads = findViewById(R.id.etThreads);
        etThreadsBatch = findViewById(R.id.etThreadsBatch);
        etUbatch = findViewById(R.id.etUbatch);
        etMaxTokens = findViewById(R.id.etMaxTokens);
        etTemp = findViewById(R.id.etTemp);
        etTopK = findViewById(R.id.etTopK);
        etTopP = findViewById(R.id.etTopP);
        etMinP = findViewById(R.id.etMinP);
        etRepeat = findViewById(R.id.etRepeat);
        etRepeatN = findViewById(R.id.etRepeatN);
        etPort = findViewById(R.id.etPort);
        etApiKey = findViewById(R.id.etApiKey);
        swLearning = findViewById(R.id.swLearning);
        swUseMemory = findViewById(R.id.swUseMemory);
        swFastGen = findViewById(R.id.swFastGen);
        swFastBatch = findViewById(R.id.swFastBatch);
        swMmap = findViewById(R.id.swMmap);
        swThinking = findViewById(R.id.swThinking);
        swLan = findViewById(R.id.swLan);
        swAutoStart = findViewById(R.id.swAutoStart);
        tgFlash = findViewById(R.id.tgFlash);
        tgKv = findViewById(R.id.tgKv);

        tvCpu.setText("Is phone me: " + CpuInfo.get().describe());
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

        btnBench.setOnClickListener(v -> runBenchmark());

        findViewById(R.id.btnCopy).setOnClickListener(v -> {
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

        findViewById(R.id.btnRocket).setOnClickListener(v -> preset(Prefs.PRESET_ROCKET, "🚀 Rocket"));
        findViewById(R.id.btnBalanced).setOnClickListener(v -> preset(Prefs.PRESET_BALANCED, "⚖️ Balanced"));
        findViewById(R.id.btnQuality).setOnClickListener(v -> preset(Prefs.PRESET_QUALITY, "🧠 Quality"));

        findViewById(R.id.btnMemView).setOnClickListener(v -> showMemories());
        findViewById(R.id.btnMemClear).setOnClickListener(v -> new AlertDialog.Builder(this)
                .setTitle("Sab bhool jaye?")
                .setMessage("Seekhi hui saari baatein delete ho jayengi.")
                .setPositiveButton("Haan", (d, w) -> {
                    memory().clear();
                    refreshMemoryCount();
                })
                .setNegativeButton("Nahi", null)
                .show());

        findViewById(R.id.btnApply).setOnClickListener(v -> applySettings());
        findViewById(R.id.btnBattery).setOnClickListener(v -> askIgnoreBatteryOptimization());

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
        refreshMemoryCount();
        if (prefs.getDownloadId() != -1) {
            ui.removeCallbacks(downloadPoll);
            ui.post(downloadPoll);
        }
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
        boolean busy = state == LlmService.State.LOADING || state == LlmService.State.GENERATING
                || state == LlmService.State.LEARNING || state == LlmService.State.BENCHMARK;
        progressStatus.setVisibility(busy ? View.VISIBLE : View.GONE);
        btnService.setText(isServiceActive() ? "Stop server" : "Start server");
        btnBench.setEnabled(state == LlmService.State.READY);
        refreshEndpoint();
        refreshModelInfo();
        refreshMemoryCount();
        if (service != null) {
            StringBuilder t = new StringBuilder();
            if (!service.getLastTiming().isEmpty()) t.append("Last: ").append(service.getLastTiming());
            if (!service.getLastBench().isEmpty()) {
                if (t.length() > 0) t.append('\n');
                t.append("Speed test: ").append(service.getLastBench());
            }
            tvTiming.setText(t.toString());
        }
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

    private void refreshModelInfo() {
        if (service == null || !service.isReady()) {
            tvModelInfo.setVisibility(View.GONE);
            return;
        }
        try {
            JSONObject o = new JSONObject(service.infoJson());
            long params = o.optLong("n_params");
            String info = String.format(Locale.US,
                    "%s • %.2fB params • model ka max context %,d • abhi %,d\n"
                            + "threads %d/%d • batch %d • KV %s • cores %s",
                    o.optString("desc"), params / 1e9, o.optInt("n_ctx_train"), o.optInt("n_ctx"),
                    o.optInt("threads"), o.optInt("threads_batch"), o.optInt("batch"),
                    o.optString("kv_type"), o.optString("cpus_gen"));
            tvModelInfo.setText(info);
            tvModelInfo.setVisibility(View.VISIBLE);
        } catch (Exception e) {
            tvModelInfo.setVisibility(View.GONE);
        }
    }

    // ------------------------------------------------------------------ benchmark

    private void runBenchmark() {
        if (service == null) {
            toast("Pehle server start karo");
            return;
        }
        btnBench.setEnabled(false);
        toast("Speed test chal raha hai (10-60 sec)…");
        service.runBenchmark((pp, tg, err) -> {
            btnBench.setEnabled(true);
            if (err != null) {
                toast(err);
                return;
            }
            new AlertDialog.Builder(this)
                    .setTitle("Speed test")
                    .setMessage(String.format(Locale.US,
                            "Prompt padhna (prefill): %.1f tok/s\nJawab likhna (generate): %.1f tok/s\n\n"
                                    + "Threads / fast cores / KV badal ke dobara test karo, jo sabse tez ho wahi rakho.",
                            pp, tg))
                    .setPositiveButton("OK", null)
                    .show();
        });
    }

    // ------------------------------------------------------------------ settings

    private void loadSettingsIntoViews() {
        etSystem.setText(prefs.getSystemPrompt());
        etMaxMem.setText(String.valueOf(prefs.getMaxMemories()));
        swLearning.setChecked(prefs.isLearning());
        swUseMemory.setChecked(prefs.isUseMemory());

        etCtx.setText(String.valueOf(prefs.getContextSize()));
        etBatch.setText(String.valueOf(prefs.getBatch()));
        etUbatch.setText(String.valueOf(prefs.getUbatch()));
        etThreads.setText(String.valueOf(prefs.getThreads()));
        etThreadsBatch.setText(String.valueOf(prefs.getThreadsBatch()));
        int fa = prefs.getFlashAttn();
        tgFlash.check(fa == 1 ? R.id.faOn : fa == 0 ? R.id.faOff : R.id.faAuto);
        int kv = prefs.getKvType();
        tgKv.check(kv == 1 ? R.id.kvQ8 : kv == 2 ? R.id.kvQ4 : R.id.kvF16);
        swFastGen.setChecked(prefs.isFastCoresGen());
        swFastBatch.setChecked(prefs.isFastCoresBatch());
        swMmap.setChecked(prefs.isMmap());

        etMaxTokens.setText(String.valueOf(prefs.getDefaultMaxTokens()));
        etTemp.setText(fmt(prefs.getTemperature()));
        etTopK.setText(String.valueOf(prefs.getTopK()));
        etTopP.setText(fmt(prefs.getTopP()));
        etMinP.setText(fmt(prefs.getMinP()));
        etRepeat.setText(fmt(prefs.getRepeatPenalty()));
        etRepeatN.setText(String.valueOf(prefs.getRepeatLastN()));
        swThinking.setChecked(prefs.isThinking());

        etPort.setText(String.valueOf(prefs.getPort()));
        etApiKey.setText(prefs.getApiKey());
        swLan.setChecked(prefs.isLanAccess());
        swAutoStart.setChecked(prefs.isAutoStart());
        refreshEndpoint();
    }

    private void preset(String name, String label) {
        prefs.applyPreset(name);
        loadSettingsIntoViews();
        toast(label + " preset laga. Ab 💾 Save & apply dabao");
    }

    private void applySettings() {
        int ctx = parseInt(etCtx, -1), batch = parseInt(etBatch, -1), ubatch = parseInt(etUbatch, -1);
        int threads = parseInt(etThreads, -1), threadsBatch = parseInt(etThreadsBatch, -1);
        int maxTok = parseInt(etMaxTokens, -1), topK = parseInt(etTopK, -1), repeatN = parseInt(etRepeatN, -1);
        int port = parseInt(etPort, -1), maxMem = parseInt(etMaxMem, -1);
        float temp = parseFloat(etTemp, -1), topP = parseFloat(etTopP, -1), minP = parseFloat(etMinP, -1);
        float repeat = parseFloat(etRepeat, -1);
        int cores = CpuInfo.get().total;

        if (ctx != 0 && (ctx < 512 || ctx > 131072)) { toast("Context 0 ya 512 se 131072"); return; }
        if (batch < 32 || batch > 4096) { toast("Batch 32 se 4096"); return; }
        if (ubatch < 32 || ubatch > batch) { toast("Ubatch 32 se batch tak"); return; }
        if (threads < 1 || threads > cores) { toast("Threads 1 se " + cores); return; }
        if (threadsBatch < 1 || threadsBatch > cores) { toast("Prompt threads 1 se " + cores); return; }
        if (maxTok < 16 || maxTok > 32768) { toast("Max tokens 16 se 32768"); return; }
        if (temp < 0 || temp > 2) { toast("Temperature 0 se 2"); return; }
        if (topK < 0 || topK > 1000) { toast("Top K 0 se 1000"); return; }
        if (topP <= 0 || topP > 1) { toast("Top P 0.01 se 1"); return; }
        if (minP < 0 || minP > 1) { toast("Min P 0 se 1"); return; }
        if (repeat < 0.5f || repeat > 2) { toast("Repeat penalty 0.5 se 2 (1 = off)"); return; }
        if (repeatN < 0 || repeatN > 2048) { toast("Repeat last N 0 se 2048"); return; }
        if (port < 1024 || port > 65535) { toast("Port 1024 se 65535"); return; }
        if (maxMem < 1 || maxMem > 500) { toast("Max yaad 1 se 500"); return; }

        int fa = tgFlash.getCheckedButtonId() == R.id.faOn ? 1
                : tgFlash.getCheckedButtonId() == R.id.faOff ? 0 : -1;
        int kv = tgKv.getCheckedButtonId() == R.id.kvQ8 ? 1
                : tgKv.getCheckedButtonId() == R.id.kvQ4 ? 2 : 0;

        prefs.saveLoadSettings(ctx, threads, threadsBatch, batch, ubatch, fa, kv,
                swMmap.isChecked(), swFastGen.isChecked(), swFastBatch.isChecked());
        prefs.saveGenSettings(maxTok, temp, topP, topK, minP, repeat, repeatN, swThinking.isChecked());
        prefs.setSystemPrompt(textOf(etSystem));
        prefs.setMaxMemories(maxMem);
        prefs.setLearning(swLearning.isChecked());
        prefs.setUseMemory(swUseMemory.isChecked());
        prefs.setPort(port);
        prefs.setApiKey(textOf(etApiKey));
        prefs.setLanAccess(swLan.isChecked());
        prefs.setAutoStart(swAutoStart.isChecked());
        refreshEndpoint();

        if (isServiceActive()) {
            LlmService.reload(this);   // load settings badli ho tabhi model reload hota hai
            toast("Save ho gaya. Server apply kar raha hai");
        } else {
            toast("Save ho gaya");
        }
    }

    // ------------------------------------------------------------------ memory

    private MemoryStore memory() {
        if (service != null) return service.memory();
        if (memoryView == null) memoryView = new MemoryStore(this);
        return memoryView;
    }

    private void refreshMemoryCount() {
        tvMemCount.setText(memory().size() + " baatein yaad");
    }

    private void showMemories() {
        List<MemoryStore.Item> items = memory().list();
        if (items.isEmpty()) {
            toast("Abhi kuch yaad nahi. Server se baat karo, khaali time me seekhega.");
            return;
        }
        String[] arr = new String[items.size()];
        for (int i = 0; i < items.size(); i++) arr[i] = "• " + items.get(i).text;
        new AlertDialog.Builder(this)
                .setTitle("Seekhi hui baatein (tap = hatao)")
                .setItems(arr, (d, which) -> new AlertDialog.Builder(this)
                        .setMessage("Ye baat hata dein?\n\n" + items.get(which).text)
                        .setPositiveButton("Hatao", (d2, w2) -> {
                            memory().remove(which);
                            refreshMemoryCount();
                        })
                        .setNegativeButton("Rehne do", null)
                        .show())
                .setPositiveButton("Band", null)
                .show();
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
                .setMessage("Gemma 4 E2B (Q4_K_M), lagbhag 3.1 GB.\nApp band kar doge tab bhi download chalta rahega.")
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

        String msg = p.message;
        ModelDownloader.cancel(this);
        refreshModelUi();
        tvModel.setText(msg.isEmpty() ? "Download fail ho gaya, dobara try karo" : msg);
        return false;
    }

    // ------------------------------------------------------------------ model: pick file

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
                 FileOutputStream os = new FileOutputStream(tmp)) {
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
                os.getFD().sync();
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
        startService();
    }

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
        findViewById(R.id.btnBattery).setVisibility(ignoring ? View.GONE : View.VISIBLE);
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

    private static int parseInt(TextInputEditText et, int def) {
        try {
            return Integer.parseInt(textOf(et));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static float parseFloat(TextInputEditText et, float def) {
        try {
            return Float.parseFloat(textOf(et));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static String fmt(float f) {
        String s = String.format(Locale.US, "%.2f", f);
        return s.endsWith("0") ? s.substring(0, s.length() - 1) : s;
    }

    private void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }
}
