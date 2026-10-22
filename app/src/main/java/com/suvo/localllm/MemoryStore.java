package com.suvo.localllm;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Self-learning memory: chats se seekhi hui baatein (user ka naam, pasand, kaam, kaise jawab chahiye...).
 * Model ke weights nahi badalte; ye baatein har request ke system prompt me jud jaati hain,
 * isliye koi bhi model "yaad" rakhta hai. File: app ke private folder me memory.json.
 */
public class MemoryStore {

    private static final String TAG = "MemoryStore";
    private static final int MAX_LEN = 200;

    public static class Item {
        public final String text;
        public final long time;

        Item(String text, long time) {
            this.text = text;
            this.time = time;
        }
    }

    private final File file;
    private final List<Item> items = new ArrayList<>();

    public MemoryStore(Context ctx) {
        file = new File(ctx.getFilesDir(), "memory.json");
        load();
    }

    public synchronized List<Item> list() {
        return new ArrayList<>(items);
    }

    public synchronized int size() {
        return items.size();
    }

    /** @return true agar nayi baat thi aur jud gayi */
    public synchronized boolean add(String text, int maxItems) {
        String t = clean(text);
        if (t.length() < 4) return false;
        String n = norm(t);
        for (Item it : items) {
            String o = norm(it.text);
            if (o.equals(n)) return false;
            if (n.length() > 15 && o.length() > 15 && (o.contains(n) || n.contains(o))) return false;
        }
        items.add(new Item(t, System.currentTimeMillis()));
        while (items.size() > Math.max(1, maxItems)) items.remove(0);   // sabse purani hatao
        save();
        return true;
    }

    public synchronized void remove(int index) {
        if (index >= 0 && index < items.size()) {
            items.remove(index);
            save();
        }
    }

    public synchronized void clear() {
        items.clear();
        save();
    }

    /** System prompt me jodne wala hissa ("" agar kuch yaad nahi) */
    public synchronized String promptBlock() {
        if (items.isEmpty()) return "";
        StringBuilder sb = new StringBuilder("\n\n[Memory: pichli baatcheet se user ke baare me ye pata hai. "
                + "Zarurat ho tab hi use karo, inhe dohraao mat.]\n");
        for (Item it : items) sb.append("- ").append(it.text).append('\n');
        return sb.toString();
    }

    /** Learner ko dikhane ke liye: pehle se kya pata hai */
    public synchronized String bulletList() {
        if (items.isEmpty()) return "(kuch nahi)";
        StringBuilder sb = new StringBuilder();
        for (Item it : items) sb.append("- ").append(it.text).append('\n');
        return sb.toString().trim();
    }

    /** Model ke output se "- fact" lines nikal ke jodo. @return kitni nayi judin */
    public int addFromModelOutput(String output, int maxItems) {
        if (output == null) return 0;
        int added = 0;
        for (String line : output.split("\n")) {
            String l = line.trim();
            if (l.isEmpty()) continue;
            if (l.toUpperCase(Locale.ROOT).startsWith("NONE")) break;
            if (!(l.startsWith("-") || l.startsWith("*") || l.startsWith("•"))) continue;
            l = l.substring(1).trim();
            if (add(l, maxItems)) added++;
        }
        return added;
    }

    // ------------------------------------------------------------------

    private static String clean(String s) {
        String t = s.replace("<thought>", "").replace("</thought>", "").replace('\n', ' ').trim();
        if (t.length() > MAX_LEN) t = t.substring(0, MAX_LEN).trim();
        return t;
    }

    private static String norm(String s) {
        return s.toLowerCase(Locale.ROOT).replaceAll("[^\\p{L}\\p{N} ]", "").replaceAll("\\s+", " ").trim();
    }

    private void load() {
        if (!file.isFile()) return;
        try (FileInputStream in = new FileInputStream(file)) {
            byte[] buf = new byte[(int) file.length()];
            int read = 0;
            while (read < buf.length) {
                int n = in.read(buf, read, buf.length - read);
                if (n < 0) break;
                read += n;
            }
            JSONArray arr = new JSONArray(new String(buf, 0, read, StandardCharsets.UTF_8));
            for (int i = 0; i < arr.length(); i++) {
                JSONObject o = arr.getJSONObject(i);
                items.add(new Item(o.optString("text"), o.optLong("time")));
            }
        } catch (Exception e) {
            Log.w(TAG, "memory load fail", e);
        }
    }

    private void save() {
        try {
            JSONArray arr = new JSONArray();
            for (Item it : items) {
                arr.put(new JSONObject().put("text", it.text).put("time", it.time));
            }
            File tmp = new File(file.getPath() + ".tmp");
            try (FileOutputStream out = new FileOutputStream(tmp)) {
                out.write(arr.toString().getBytes(StandardCharsets.UTF_8));
                out.getFD().sync();
            }
            //noinspection ResultOfMethodCallIgnored
            tmp.renameTo(file);
        } catch (Exception e) {
            Log.w(TAG, "memory save fail", e);
        }
    }
}
