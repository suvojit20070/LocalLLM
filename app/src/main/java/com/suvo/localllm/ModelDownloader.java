package com.suvo.localllm;

import android.app.DownloadManager;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;

import java.io.File;

/**
 * Android DownloadManager se model download.
 * App band ho jaye tab bhi download chalta rehta hai, network tootne pe khud resume hota hai.
 */
public final class ModelDownloader {

    public static final int STATUS_NONE = 0;
    public static final int STATUS_RUNNING = 1;
    public static final int STATUS_DONE = 2;
    public static final int STATUS_FAILED = 3;

    public static class Progress {
        public int status = STATUS_NONE;
        public long downloaded;
        public long total;
        public String message = "";

        public int percent() {
            return total > 0 ? (int) (downloaded * 100 / total) : 0;
        }
    }

    private ModelDownloader() {}

    public static File targetFile(Context ctx) {
        return new File(Prefs.modelDir(ctx), Prefs.MODEL_FILE);
    }

    /** 1-click download shuru. Purana adhura download cancel ho jata hai. */
    public static void start(Context ctx) {
        Prefs prefs = new Prefs(ctx);
        DownloadManager dm = (DownloadManager) ctx.getSystemService(Context.DOWNLOAD_SERVICE);

        long old = prefs.getDownloadId();
        if (old != -1) dm.remove(old);

        File target = targetFile(ctx);
        //noinspection ResultOfMethodCallIgnored
        target.delete();

        DownloadManager.Request req = new DownloadManager.Request(Uri.parse(Prefs.MODEL_URL))
                .setTitle("Gemma 4 E2B model")
                .setDescription("Local LLM ke liye model download ho raha hai")
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                .setDestinationInExternalFilesDir(ctx, "models", Prefs.MODEL_FILE)
                .setAllowedOverMetered(true)
                .setAllowedOverRoaming(true);

        prefs.setDownloadId(dm.enqueue(req));
    }

    public static void cancel(Context ctx) {
        Prefs prefs = new Prefs(ctx);
        long id = prefs.getDownloadId();
        if (id != -1) {
            DownloadManager dm = (DownloadManager) ctx.getSystemService(Context.DOWNLOAD_SERVICE);
            dm.remove(id);
        }
        prefs.setDownloadId(-1);
        //noinspection ResultOfMethodCallIgnored
        targetFile(ctx).delete();
    }

    public static Progress query(Context ctx) {
        Progress p = new Progress();
        long id = new Prefs(ctx).getDownloadId();
        if (id == -1) return p;

        DownloadManager dm = (DownloadManager) ctx.getSystemService(Context.DOWNLOAD_SERVICE);
        try (Cursor c = dm.query(new DownloadManager.Query().setFilterById(id))) {
            if (c == null || !c.moveToFirst()) {
                p.status = STATUS_FAILED;
                p.message = "Download cancel ho gaya";
                return p;
            }
            int st = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS));
            p.downloaded = c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR));
            p.total = c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES));
            int reason = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON));

            switch (st) {
                case DownloadManager.STATUS_SUCCESSFUL:
                    p.status = STATUS_DONE;
                    break;
                case DownloadManager.STATUS_FAILED:
                    p.status = STATUS_FAILED;
                    p.message = "Download fail (code " + reason + ")";
                    break;
                case DownloadManager.STATUS_PAUSED:
                    p.status = STATUS_RUNNING;
                    p.message = "Ruka hua (network ka wait)";
                    break;
                default:
                    p.status = STATUS_RUNNING;
                    p.message = st == DownloadManager.STATUS_PENDING ? "Shuru ho raha hai…" : "";
            }
        }
        return p;
    }

    /**
     * Download complete hone pe model set karo. Activity aur DownloadReceiver dono call karte hain.
     * @return true agar model ab ready hai
     */
    public static boolean finishIfDone(Context ctx) {
        Prefs prefs = new Prefs(ctx);
        if (prefs.getDownloadId() == -1) return false;
        Progress p = query(ctx);
        if (p.status == STATUS_DONE) {
            File f = targetFile(ctx);
            prefs.setDownloadId(-1);
            if (f.isFile() && f.length() > 100L * 1024 * 1024) {
                prefs.setModelPath(f.getAbsolutePath());
                return true;
            }
        } else if (p.status == STATUS_FAILED) {
            prefs.setDownloadId(-1);
        }
        return false;
    }
}
