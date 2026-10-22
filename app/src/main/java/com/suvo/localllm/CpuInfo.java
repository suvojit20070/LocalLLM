package com.suvo.localllm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Phone ke CPU cores pehchaanta hai.
 * Mobile chips me fast (big/prime) aur slow (little) cores hote hain. LLM generation me
 * slow cores ko mila doge to fast cores unka wait karte hain, isliye sirf fast cores pe
 * threads chalana aksar zyada tez hota hai.
 */
public final class CpuInfo {

    public final int total;
    public final int[] maxFreqKhz;     // har core ki max speed
    public final int[] fastCores;      // little cores ko chhod ke baaki
    public final int[] allCores;

    private static CpuInfo cached;

    public static synchronized CpuInfo get() {
        if (cached == null) cached = new CpuInfo();
        return cached;
    }

    private CpuInfo() {
        int n = Runtime.getRuntime().availableProcessors();
        int present = countPresent();
        if (present > n) n = present;
        total = n;
        maxFreqKhz = new int[n];
        int min = Integer.MAX_VALUE, max = 0;
        for (int i = 0; i < n; i++) {
            maxFreqKhz[i] = readInt("/sys/devices/system/cpu/cpu" + i + "/cpufreq/cpuinfo_max_freq");
            if (maxFreqKhz[i] > 0) {
                min = Math.min(min, maxFreqKhz[i]);
                max = Math.max(max, maxFreqKhz[i]);
            }
        }

        allCores = new int[n];
        for (int i = 0; i < n; i++) allCores[i] = i;

        List<Integer> fast = new ArrayList<>();
        if (max > 0 && min < max) {
            for (int i = 0; i < n; i++) if (maxFreqKhz[i] > min) fast.add(i);
        }
        if (fast.isEmpty()) {
            fastCores = allCores.clone();
        } else {
            fastCores = new int[fast.size()];
            for (int i = 0; i < fast.size(); i++) fastCores[i] = fast.get(i);
        }
    }

    public int fastCount() { return fastCores.length; }

    public int slowCount() { return total - fastCores.length; }

    /** Generation ke liye suggested threads */
    public int recommendedThreads() {
        return Math.max(1, Math.min(fastCores.length, 4));
    }

    /** Prefill (prompt padhna) ke liye: zyada threads se fayda hota hai */
    public int recommendedBatchThreads() {
        return Math.max(2, Math.min(total, 8));
    }

    public String describe() {
        StringBuilder sb = new StringBuilder();
        sb.append(total).append(" cores: ");
        if (slowCount() > 0) {
            sb.append(fastCount()).append(" fast + ").append(slowCount()).append(" slow");
        } else {
            sb.append("sab barabar");
        }
        int top = 0;
        for (int f : maxFreqKhz) top = Math.max(top, f);
        if (top > 0) sb.append(String.format(Locale.US, " (max %.2f GHz)", top / 1_000_000.0));
        return sb.toString();
    }

    private static int countPresent() {
        int c = 0;
        while (new File("/sys/devices/system/cpu/cpu" + c).exists()) c++;
        return c;
    }

    private static int readInt(String path) {
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String line = r.readLine();
            return line == null ? 0 : Integer.parseInt(line.trim());
        } catch (Exception e) {
            return 0;
        }
    }
}
