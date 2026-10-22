package com.suvo.localllm;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

/** Phone restart / app update ke baad service apne aap start karta hai. */
public class BootReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
        String a = intent.getAction();
        if (!Intent.ACTION_BOOT_COMPLETED.equals(a)
                && !"android.intent.action.QUICKBOOT_POWERON".equals(a)
                && !Intent.ACTION_MY_PACKAGE_REPLACED.equals(a)) {
            return;
        }
        Prefs prefs = new Prefs(context);
        if (prefs.isAutoStart() && prefs.isServiceEnabled() && prefs.hasModel()) {
            try {
                LlmService.start(context);
            } catch (Exception ignored) {
                // kuch ROMs boot pe foreground service block karte hain
            }
        }
    }
}
