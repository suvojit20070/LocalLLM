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
 *   GET  /health
 *   GET  /v1/models
 *   POST /v1/chat/completions   (stream true/false dono)
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
        String getSystemPrompt();
        int getDefaultMaxTokens();
        Completion complete(String[] roles, String[] contents, GenParams params, LlamaBridge.TokenCallback cb);
    }

    public static class GenParams {
        public int maxTokens;
        public float temperature = 0.7f;
        public float topP = 0.95f;
        public int topK = 40;
        public float frequencyPenalty = 0f;
        public float presencePenalty = 0f;
    }

    public static class Completion {
        public String text = "";
        public String error;
        public int promptTokens, completionTokens, reusedTokens;
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
    private Thread acceptThread;
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

        acceptThread = new Thread(() -> {
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

            // CORS (browser based clients ke liye)
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

            switch (path) {
                case "/v1/models":
                case "/models":
                    if (!"GET".equals(req.method)) { sendError(out, 405, "Method not allowed", "invalid_request_error"); return; }
                    sendJson(out, 200, modelsJson());
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
                //noinspection StatementWithEmptyBody
                while (true) { String t = readLine(in); if (t == null || t.isEmpty()) break; }
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
            readLine(in); // chunk ke baad CRLF
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

        // Request me system prompt nahi hai to app wala system prompt lagao
        String sys = engine.getSystemPrompt();
        boolean hasSystem = "system".equals(roles.get(0)) || "developer".equals(roles.get(0));
        if (!hasSystem && sys != null && !sys.trim().isEmpty()) {
            roles.add(0, "system");
            contents.add(0, sys);
        }

        // ---- params ----
        GenParams p = new GenParams();
        p.maxTokens = body.optInt("max_completion_tokens", body.optInt("max_tokens", engine.getDefaultMaxTokens()));
        p.temperature = (float) body.optDouble("temperature", 0.7);
        p.topP = (float) body.optDouble("top_p", 0.95);
        p.topK = body.optInt("top_k", 40);
        p.frequencyPenalty = (float) body.optDouble("frequency_penalty", 0);
        p.presencePenalty = (float) body.optDouble("presence_penalty", 0);
        boolean stream = body.optBoolean("stream", false);
        JSONObject so = body.optJSONObject("stream_options");
        boolean includeUsage = so != null && so.optBoolean("include_usage", false);

        if (!engine.isReady()) {
            sendError(out, 503, "Model ready nahi hai: " + engine.getStatusMessage(), "server_error");
            return;
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
                    .put("usage", usageJson(res));
            sendJson(out, 200, o);
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

        // pehla chunk: role
        writeSse(out, chunkJson(id, created, model, new JSONObject().put("role", "assistant").put("content", ""), null));

        Completion res = engine.complete(r, c, p, utf8 -> {
            try {
                String piece = new String(utf8, StandardCharsets.UTF_8);
                writeSse(out, chunkJson(id, created, model, new JSONObject().put("content", piece), null));
            } catch (Exception e) {
                // client gaya -> exception native ko generation rokne ka signal deta hai
                throw new RuntimeException(e);
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
                            .put("usage", usageJson(res));
                    writeSse(out, u.toString());
                }
            }
            writeSse(out, "[DONE]");
            writeChunk(out, new byte[0]); // chunked stream ka end
        } catch (IOException ignored) {
            // client already chala gaya
        }
    }

    // ================================================================ helpers

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
                + "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
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
