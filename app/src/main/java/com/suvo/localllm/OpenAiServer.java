package com.suvo.localllm;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Chhota sa OpenAI-compatible HTTP server (koi library nahi).
 *
 * Endpoints:
 *   GET    /health
 *   GET    /v1/models
 *   GET    /v1/info                 model + speed settings + last request timing
 *   POST   /v1/chat/completions     (stream true/false dono)
 *   GET    /v1/memory               seekhi hui baatein
 *   POST   /v1/memory               {"text": "..."} khud jodo
 *   DELETE /v1/memory               sab bhool jao
 *   DELETE /v1/memory/{index}       ek baat hatao
 *
 * Ek time pe ek hi reply generate hota hai, baaki requests line me wait karti hain.
 */
public class OpenAiServer {

    private static final String TAG = "OpenAiServer";
    private static final int MAX_BODY = 16 * 1024 * 1024;

    /** LlmService ye implement karta hai. */
    public interface Engine {
        boolean isReady();
        String getStatusMessage();
        String getModelId();
        /** Request ka system prompt (ya null) le ke final system prompt (app default + memory) */
        String buildSystemPrompt(String requestSystem, boolean useMemory);
        GenParams defaultParams();
        Completion complete(String[] roles, String[] contents, GenParams params, LlamaBridge.TokenCallback cb);
        /** Ek user/assistant baatcheet poori hui, learning ke liye */
        void onExchange(String userText, String reply);
        String infoJson();
        MemoryStore memory();
        int maxMemories();
    }

    public static class GenParams {
        public int maxTokens = 1024;
        public float temperature = 0.7f;
        public float topP = 0.95f;
        public int topK = 40;
        public float minP = 0.05f;
        public float repeatPenalty = 1.1f;
        public int repeatLastN = 64;
        public float frequencyPenalty = 0f;
        public float presencePenalty = 0f;
        public int seed = -1;
        public boolean thinking = false;
        public boolean learn = true;
        public boolean useMemory = true;
        public boolean internal = false;   // learner jaise andar ke kaam (stats/learning me nahi ginte)
    }

    public static class Completion {
        public String text = "";
        public String error;
        public int promptTokens, completionTokens, reusedTokens;
        public int prefillMs, genMs;
        public int finish = LlamaBridge.FINISH_STOP;

        public static Completion error(String e) {
            Completion c = new Completion();
            c.error = e;
            return c;
        }

        String finishReason() {
            return finish == LlamaBridge.FINISH_LENGTH ? "length" : "stop";
        }
    }

    private final Engine engine;
    private final int port;
    private final boolean lan;
    private final String apiKey;

    private ServerSocket serverSocket;
    private ExecutorService clients;
    private volatile boolean running;

    public OpenAiServer(Engine engine, int port, boolean lan, String apiKey) {
        this.engine = engine;
        this.port = port;
        this.lan = lan;
        this.apiKey = apiKey == null ? "" : apiKey.trim();
    }

    /** @throws IOException agar port busy ho */
    public synchronized void start() throws IOException {
        if (running) return;
        ServerSocket ss = new ServerSocket();
        ss.setReuseAddress(true);
        InetAddress addr = lan ? InetAddress.getByName("0.0.0.0") : InetAddress.getByName("127.0.0.1");
        ss.bind(new InetSocketAddress(addr, port), 50);
        serverSocket = ss;
        clients = Executors.newFixedThreadPool(8);
        running = true;

        Thread acceptThread = new Thread(() -> {
            while (running) {
                try {
                    Socket s = serverSocket.accept();
                    clients.execute(() -> handle(s));
                } catch (IOException e) {
                    if (running) Log.w(TAG, "accept fail", e);
                }
            }
        }, "openai-accept");
        acceptThread.start();
        Log.i(TAG, "Server chalu: port " + port + (lan ? " (LAN)" : " (localhost)"));
    }

    public synchronized void stop() {
        running = false;
        try {
            if (serverSocket != null) serverSocket.close();
        } catch (IOException ignored) { }
        if (clients != null) clients.shutdownNow();
        serverSocket = null;
    }

    public boolean isRunning() { return running; }

    public int getPort() { return port; }

    // ================================================================ HTTP

    private static class Request {
        String method, path;
        final Map<String, String> headers = new HashMap<>();
        byte[] body = new byte[0];
    }

    private void handle(Socket socket) {
        try (Socket s = socket) {
            s.setSoTimeout(60_000);
            s.setTcpNoDelay(true);
            InputStream in = new BufferedInputStream(s.getInputStream());
            OutputStream out = s.getOutputStream();

            Request req = readRequest(in);
            if (req == null) return;

            if ("OPTIONS".equals(req.method)) {
                sendRaw(out, 204, "text/plain", new byte[0]);
                return;
            }

            String path = req.path;
            int q = path.indexOf('?');
            if (q >= 0) path = path.substring(0, q);
            if (path.length() > 1 && path.endsWith("/")) path = path.substring(0, path.length() - 1);

            if (path.equals("/") || path.equals("/health")) {
                JSONObject o = new JSONObject();
                o.put("status", engine.isReady() ? "ok" : "loading");
                o.put("model_ready", engine.isReady());
                o.put("message", engine.getStatusMessage());
                sendJson(out, 200, o);
                return;
            }

            if (!checkAuth(req)) {
                sendError(out, 401, "Invalid API key", "invalid_request_error");
                return;
            }

            if (path.startsWith("/v1/memory") || path.startsWith("/memory")) {
                memoryEndpoint(req, path, out);
                return;
            }

            switch (path) {
                case "/v1/models":
                case "/models":
                    sendJson(out, 200, modelsJson());
                    break;

                case "/v1/info":
                case "/info":
                    sendJson(out, 200, new JSONObject(engine.infoJson()));
                    break;

                case "/v1/chat/completions":
                case "/chat/completions":
                    if (!"POST".equals(req.method)) { sendError(out, 405, "Method not allowed", "invalid_request_error"); return; }
                    chatCompletions(req, out);
                    break;

                default:
                    sendError(out, 404, "Unknown endpoint: " + path, "invalid_request_error");
            }
        } catch (SocketException ignored) {
            // client ne connection tod diya
        } catch (Exception e) {
            Log.w(TAG, "request fail", e);
        }
    }

    private Request readRequest(InputStream in) throws IOException {
        String line = readLine(in);
        if (line == null || line.isEmpty()) return null;
        String[] parts = line.split(" ");
        if (parts.length < 2) return null;

        Request r = new Request();
        r.method = parts[0].toUpperCase(Locale.ROOT);
        r.path = parts[1];

        String h;
        while ((h = readLine(in)) != null && !h.isEmpty()) {
            int c = h.indexOf(':');
            if (c > 0) r.headers.put(h.substring(0, c).trim().toLowerCase(Locale.ROOT), h.substring(c + 1).trim());
        }

        String cl = r.headers.get("content-length");
        if (cl != null) {
            int len = Integer.parseInt(cl.trim());
            if (len < 0 || len > MAX_BODY) throw new IOException("Body bahut badi hai");
            byte[] body = new byte[len];
            int read = 0;
            while (read < len) {
                int n = in.read(body, read, len - read);
                if (n < 0) break;
                read += n;
            }
            r.body = body;
        } else if ("chunked".equalsIgnoreCase(r.headers.get("transfer-encoding"))) {
            r.body = readChunkedBody(in);
        }
        return r;
    }

    private static byte[] readChunkedBody(InputStream in) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        while (true) {
            String sizeLine = readLine(in);
            if (sizeLine == null) break;
            int semi = sizeLine.indexOf(';');
            if (semi >= 0) sizeLine = sizeLine.substring(0, semi);
            int size = Integer.parseInt(sizeLine.trim(), 16);
            if (size == 0) {
                while (true) {
                    String t = readLine(in);
                    if (t == null || t.isEmpty()) break;
                }
                break;
            }
            if (bos.size() + size > MAX_BODY) throw new IOException("Body bahut badi hai");
            byte[] buf = new byte[size];
            int read = 0;
            while (read < size) {
                int n = in.read(buf, read, size - read);
                if (n < 0) throw new IOException("Adhura chunk");
                read += n;
            }
            bos.write(buf);
            readLine(in);
        }
        return bos.toByteArray();
    }

    private static String readLine(InputStream in) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        int b;
        while ((b = in.read()) != -1) {
            if (b == '\n') break;
            if (b != '\r') bos.write(b);
            if (bos.size() > 16 * 1024) throw new IOException("Header line bahut lambi");
        }
        if (b == -1 && bos.size() == 0) return null;
        return bos.toString("ISO-8859-1");
    }

    private boolean checkAuth(Request req) {
        if (apiKey.isEmpty()) return true;
        String auth = req.headers.get("authorization");
        if (auth != null && auth.regionMatches(true, 0, "Bearer ", 0, 7)) {
            return apiKey.equals(auth.substring(7).trim());
        }
        String x = req.headers.get("x-api-key");
        return apiKey.equals(x);
    }

    // ================================================================ endpoints

    private JSONObject modelsJson() throws JSONException {
        JSONObject m = new JSONObject();
        m.put("id", engine.getModelId());
        m.put("object", "model");
        m.put("created", System.currentTimeMillis() / 1000);
        m.put("owned_by", "local");
        JSONObject o = new JSONObject();
        o.put("object", "list");
        o.put("data", new JSONArray().put(m));
        return o;
    }

    private void memoryEndpoint(Request req, String path, OutputStream out) throws IOException, JSONException {
        MemoryStore mem = engine.memory();
        String rest = path.substring(path.indexOf("memory") + 6);   // "" ya "/3"

        if ("GET".equals(req.method)) {
            JSONArray arr = new JSONArray();
            List<MemoryStore.Item> items = mem.list();
            for (int i = 0; i < items.size(); i++) {
                arr.put(new JSONObject().put("index", i).put("text", items.get(i).text)
                        .put("time", items.get(i).time));
            }
            sendJson(out, 200, new JSONObject().put("count", items.size()).put("data", arr));
        } else if ("POST".equals(req.method)) {
            JSONObject body = new JSONObject(new String(req.body, StandardCharsets.UTF_8));
            boolean added = mem.add(body.optString("text", ""), engine.maxMemories());
            sendJson(out, 200, new JSONObject().put("added", added).put("count", mem.size()));
        } else if ("DELETE".equals(req.method)) {
            if (rest.startsWith("/") && rest.length() > 1) {
                try {
                    mem.remove(Integer.parseInt(rest.substring(1)));
                } catch (NumberFormatException e) {
                    sendError(out, 400, "Index number hona chahiye", "invalid_request_error");
                    return;
                }
            } else {
                mem.clear();
            }
            sendJson(out, 200, new JSONObject().put("count", mem.size()));
        } else {
            sendError(out, 405, "Method not allowed", "invalid_request_error");
        }
    }

    private void chatCompletions(Request req, OutputStream out) throws IOException, JSONException {
        JSONObject body;
        try {
            body = new JSONObject(new String(req.body, StandardCharsets.UTF_8));
        } catch (JSONException e) {
            sendError(out, 400, "Invalid JSON body", "invalid_request_error");
            return;
        }

        JSONArray messages = body.optJSONArray("messages");
        if (messages == null || messages.length() == 0) {
            sendError(out, 400, "'messages' required", "invalid_request_error");
            return;
        }

        // ---- messages -> roles/contents ----
        List<String> roles = new ArrayList<>();
        List<String> contents = new ArrayList<>();
        for (int i = 0; i < messages.length(); i++) {
            JSONObject m = messages.optJSONObject(i);
            if (m == null) continue;
            String role = m.optString("role", "user");
            String content = contentToText(m.opt("content"));
            if ("tool".equals(role) || "function".equals(role)) {
                content = "[Tool result] " + content;
            }
            roles.add(role);
            contents.add(content);
        }
        if (roles.isEmpty()) {
            sendError(out, 400, "'messages' khaali hai", "invalid_request_error");
            return;
        }

        // ---- params (request me na ho to app ki setting) ----
        GenParams p = engine.defaultParams();
        p.maxTokens = body.optInt("max_completion_tokens", body.optInt("max_tokens", p.maxTokens));
        p.temperature = (float) body.optDouble("temperature", p.temperature);
        p.topP = (float) body.optDouble("top_p", p.topP);
        p.topK = body.optInt("top_k", p.topK);
        p.minP = (float) body.optDouble("min_p", p.minP);
        p.repeatPenalty = (float) body.optDouble("repeat_penalty", p.repeatPenalty);
        p.repeatLastN = body.optInt("repeat_last_n", p.repeatLastN);
        p.frequencyPenalty = (float) body.optDouble("frequency_penalty", p.frequencyPenalty);
        p.presencePenalty = (float) body.optDouble("presence_penalty", p.presencePenalty);
        p.seed = body.optInt("seed", p.seed);
        p.thinking = parseThinking(body, p.thinking);
        p.learn = body.optBoolean("learn", p.learn);
        p.useMemory = body.optBoolean("memory", p.useMemory);
        boolean stream = body.optBoolean("stream", false);
        JSONObject so = body.optJSONObject("stream_options");
        boolean includeUsage = so != null && so.optBoolean("include_usage", false);

        // ---- system prompt (app default + memory) ----
        boolean hasSystem = "system".equals(roles.get(0)) || "developer".equals(roles.get(0));
        String sys = engine.buildSystemPrompt(hasSystem ? contents.get(0) : null, p.useMemory);
        if (hasSystem) {
            contents.set(0, sys);
        } else if (sys != null && !sys.trim().isEmpty()) {
            roles.add(0, "system");
            contents.add(0, sys);
        }

        if (!engine.isReady()) {
            sendError(out, 503, "Model ready nahi hai: " + engine.getStatusMessage(), "server_error");
            return;
        }

        String lastUser = "";
        for (int i = roles.size() - 1; i >= 0; i--) {
            if ("user".equals(roles.get(i))) { lastUser = contents.get(i); break; }
        }

        String[] r = roles.toArray(new String[0]);
        String[] c = contents.toArray(new String[0]);
        String id = "chatcmpl-" + UUID.randomUUID().toString().replace("-", "").substring(0, 24);
        long created = System.currentTimeMillis() / 1000;
        String model = engine.getModelId();

        if (!stream) {
            Completion res = engine.complete(r, c, p, null);
            if (res.error != null) {
                sendError(out, 500, res.error, "server_error");
                return;
            }
            JSONObject msg = new JSONObject().put("role", "assistant").put("content", res.text);
            JSONObject choice = new JSONObject()
                    .put("index", 0)
                    .put("message", msg)
                    .put("logprobs", JSONObject.NULL)
                    .put("finish_reason", res.finishReason());
            JSONObject o = new JSONObject()
                    .put("id", id)
                    .put("object", "chat.completion")
                    .put("created", created)
                    .put("model", model)
                    .put("choices", new JSONArray().put(choice))
                    .put("usage", usageJson(res))
                    .put("timings", timingsJson(res));
            sendJson(out, 200, o);
            if (p.learn) engine.onExchange(lastUser, res.text);
            return;
        }

        // ---- streaming (SSE, chunked) ----
        String head = "HTTP/1.1 200 OK\r\n"
                + "Content-Type: text/event-stream; charset=utf-8\r\n"
                + "Cache-Control: no-cache\r\n"
                + "Access-Control-Allow-Origin: *\r\n"
                + "Connection: close\r\n"
                + "Transfer-Encoding: chunked\r\n\r\n";
        out.write(head.getBytes(StandardCharsets.ISO_8859_1));
        out.flush();

        writeSse(out, chunkJson(id, created, model, new JSONObject().put("role", "assistant").put("content", ""), null));

        Completion res = engine.complete(r, c, p, utf8 -> {
            try {
                String piece = new String(utf8, StandardCharsets.UTF_8);
                writeSse(out, chunkJson(id, created, model, new JSONObject().put("content", piece), null));
            } catch (Exception e) {
                throw new RuntimeException(e);   // client gaya -> native generation rok deta hai
            }
        });

        try {
            if (res.error != null) {
                JSONObject err = new JSONObject().put("error",
                        new JSONObject().put("message", res.error).put("type", "server_error"));
                writeSse(out, err.toString());
            } else {
                writeSse(out, chunkJson(id, created, model, new JSONObject(), res.finishReason()));
                if (includeUsage) {
                    JSONObject u = new JSONObject()
                            .put("id", id)
                            .put("object", "chat.completion.chunk")
                            .put("created", created)
                            .put("model", model)
                            .put("choices", new JSONArray())
                            .put("usage", usageJson(res))
                            .put("timings", timingsJson(res));
                    writeSse(out, u.toString());
                }
            }
            writeSse(out, "[DONE]");
            writeChunk(out, new byte[0]);
        } catch (IOException ignored) {
            // client already chala gaya
        }
        if (res.error == null && p.learn && res.finish != LlamaBridge.FINISH_CANCELLED) {
            engine.onExchange(lastUser, res.text);
        }
    }

    // ================================================================ helpers

    /**
     * Thinking on/off request se:
     *   "thinking": true/false
     *   "reasoning_effort": "none" = off, baaki = on
     *   "chat_template_kwargs": {"enable_thinking": false}
     */
    private static boolean parseThinking(JSONObject body, boolean def) {
        Object t = body.opt("thinking");
        if (t instanceof Boolean) return (Boolean) t;
        if (t instanceof JSONObject) {
            String type = ((JSONObject) t).optString("type", "");
            if (!type.isEmpty()) return !"disabled".equals(type);
        }
        String effort = body.optString("reasoning_effort", "");
        if (!effort.isEmpty()) return !"none".equalsIgnoreCase(effort);
        JSONObject kw = body.optJSONObject("chat_template_kwargs");
        if (kw != null && kw.has("enable_thinking")) return kw.optBoolean("enable_thinking", def);
        return def;
    }

    /** content string ho ya [{type:"text", text:"..."}] array, dono handle */
    private static String contentToText(Object content) {
        if (content == null || content == JSONObject.NULL) return "";
        if (content instanceof String) return (String) content;
        if (content instanceof JSONArray) {
            JSONArray arr = (JSONArray) content;
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < arr.length(); i++) {
                Object item = arr.opt(i);
                if (item instanceof JSONObject) {
                    JSONObject o = (JSONObject) item;
                    if ("text".equals(o.optString("type")) || o.has("text")) {
                        if (sb.length() > 0) sb.append('\n');
                        sb.append(o.optString("text"));
                    }
                } else if (item instanceof String) {
                    if (sb.length() > 0) sb.append('\n');
                    sb.append(item);
                }
            }
            return sb.toString();
        }
        return String.valueOf(content);
    }

    private static String chunkJson(String id, long created, String model, JSONObject delta, String finish)
            throws JSONException {
        JSONObject choice = new JSONObject()
                .put("index", 0)
                .put("delta", delta)
                .put("logprobs", JSONObject.NULL)
                .put("finish_reason", finish == null ? JSONObject.NULL : finish);
        return new JSONObject()
                .put("id", id)
                .put("object", "chat.completion.chunk")
                .put("created", created)
                .put("model", model)
                .put("choices", new JSONArray().put(choice))
                .toString();
    }

    private static JSONObject usageJson(Completion res) throws JSONException {
        return new JSONObject()
                .put("prompt_tokens", res.promptTokens)
                .put("completion_tokens", res.completionTokens)
                .put("total_tokens", res.promptTokens + res.completionTokens)
                .put("prompt_tokens_details", new JSONObject().put("cached_tokens", res.reusedTokens));
    }

    /** llama.cpp server jaisa timings block: speed check karne ke liye */
    private static JSONObject timingsJson(Completion res) throws JSONException {
        int processed = Math.max(0, res.promptTokens - res.reusedTokens);
        return new JSONObject()
                .put("cache_n", res.reusedTokens)
                .put("prompt_n", processed)
                .put("prompt_ms", res.prefillMs)
                .put("prompt_per_second", res.prefillMs > 0 ? processed * 1000.0 / res.prefillMs : 0)
                .put("predicted_n", res.completionTokens)
                .put("predicted_ms", res.genMs)
                .put("predicted_per_second", res.genMs > 0 ? res.completionTokens * 1000.0 / res.genMs : 0);
    }

    private static void writeSse(OutputStream out, String data) throws IOException {
        writeChunk(out, ("data: " + data + "\n\n").getBytes(StandardCharsets.UTF_8));
    }

    private static void writeChunk(OutputStream out, byte[] data) throws IOException {
        out.write((Integer.toHexString(data.length) + "\r\n").getBytes(StandardCharsets.ISO_8859_1));
        out.write(data);
        out.write("\r\n".getBytes(StandardCharsets.ISO_8859_1));
        out.flush();
    }

    private static void sendError(OutputStream out, int code, String message, String type) throws IOException {
        try {
            JSONObject o = new JSONObject().put("error",
                    new JSONObject().put("message", message).put("type", type).put("code", code));
            sendJson(out, code, o);
        } catch (JSONException e) {
            throw new IOException(e);
        }
    }

    private static void sendJson(OutputStream out, int code, JSONObject o) throws IOException {
        sendRaw(out, code, "application/json; charset=utf-8", o.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void sendRaw(OutputStream out, int code, String type, byte[] body) throws IOException {
        String head = "HTTP/1.1 " + code + " " + statusText(code) + "\r\n"
                + "Content-Type: " + type + "\r\n"
                + "Content-Length: " + body.length + "\r\n"
                + "Access-Control-Allow-Origin: *\r\n"
                + "Access-Control-Allow-Headers: *\r\n"
                + "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
                + "Connection: close\r\n\r\n";
        out.write(head.getBytes(StandardCharsets.ISO_8859_1));
        out.write(body);
        out.flush();
    }

    private static String statusText(int code) {
        switch (code) {
            case 200: return "OK";
            case 204: return "No Content";
            case 400: return "Bad Request";
            case 401: return "Unauthorized";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 503: return "Service Unavailable";
            default:  return "Internal Server Error";
        }
    }
}
