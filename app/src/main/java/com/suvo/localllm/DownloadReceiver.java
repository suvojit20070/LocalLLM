package com.suvo.localllm;

import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

/** Download khatam hote hi model set karke service start (app band ho tab bhi). */
public class DownloadReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
        if (!DownloadManager.ACTION_DOWNLOAD_COMPLETE.equals(intent.getAction())) return;
        long id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1);
        Prefs prefs = new Prefs(context);
        if (id == -1 || id != prefs.getDownloadId()) return;

        if (ModelDownloader.finishIfDone(context) && prefs.isServiceEnabled()) {
            try {
                LlmService.start(context);
            } catch (Exception ignored) {
                // Android 12+ background se foreground service start allow nahi karta.
                // Koi baat nahi: app kholte hi service start ho jayegi.
            }
        }
    }
}
