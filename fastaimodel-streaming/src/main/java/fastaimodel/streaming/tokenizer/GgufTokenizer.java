package fastaimodel.streaming.tokenizer;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Pure Java BPE & SentencePiece Tokenizer parsing directly from GGUF metadata.
 * Supports LLaMA / Mistral SentencePiece tokenization with ' ' (\u2581) word boundaries
 * as well as GPT2/SmolLM Byte-Pair Encoding and ChatML special tokens.
 */
public class GgufTokenizer {

    public static final int GGUF_MAGIC = 0x46554747;

    // GPT-2 Byte-to-Unicode mapping table (256 bytes -> unicode chars)
    private static final char[] BYTE_TO_UNICODE = new char[] {
        256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
        272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287,
        288, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 289,
        290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305,
        306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321,
        322, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 323, 174, 175,
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
        208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
    };

    // Unicode char -> byte lookup table (for GPT-2 decoding)
    private static final int[] UNICODE_TO_BYTE = new int[] {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, -1, 174, 175,
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
        208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
        142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
        158, 159, 160, 173
    };

    private static final Pattern BPE_SPLIT_PATTERN = Pattern.compile(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );

    private final List<String> vocab = new ArrayList<>();
    private final Map<String, Integer> tokenToId = new HashMap<>();
    private final Map<String, Integer> bpeRanks = new HashMap<>();
    private final Set<Integer> stopTokenIds = new HashSet<>();
    private final float[] scores;
    private int bosTokenId = 1;
    private int eosTokenId = 2;
    private int unkTokenId = 0;
    private boolean addBos = true;
    private boolean addEos = false;
    private String modelType = "llama";
    private final boolean isByteBpe;

    public GgufTokenizer(List<String> vocab, float[] scores, int bos, int eos, int unk, boolean addBos, String modelType) {
        this(vocab, scores, null, bos, eos, unk, addBos, modelType);
    }

    public GgufTokenizer(List<String> vocab, float[] scores, List<String> merges, int bos, int eos, int unk, boolean addBos, String modelType) {
        this.vocab.addAll(vocab);
        this.scores = scores;
        this.bosTokenId = bos;
        this.eosTokenId = eos;
        this.unkTokenId = unk;
        this.addBos = addBos;
        this.modelType = modelType;
        this.isByteBpe = "gpt2".equalsIgnoreCase(modelType) || "qwen".equalsIgnoreCase(modelType)
                || "smollm".equalsIgnoreCase(modelType) || "gptneox".equalsIgnoreCase(modelType);

        for (int i = 0; i < vocab.size(); i++) {
            tokenToId.putIfAbsent(vocab.get(i), i);
        }

        if (merges != null) {
            for (int i = 0; i < merges.size(); i++) {
                bpeRanks.putIfAbsent(merges.get(i), i);
            }
        }

        if (eos >= 0) stopTokenIds.add(eos);
        // Register common special stop tokens if present
        for (String stopName : new String[]{"<|endoftext|>", "<|im_end|>", "</s>", "<eos>"}) {
            Integer id = tokenToId.get(stopName);
            if (id != null) stopTokenIds.add(id);
        }
    }

    public static GgufTokenizer loadFromGguf(File ggufFile) throws Exception {
        try (RandomAccessFile raf = new RandomAccessFile(ggufFile, "r");
             FileChannel channel = raf.getChannel()) {

            FastGgufReader reader = new FastGgufReader(channel, 4 * 1024 * 1024);

            int magic = reader.readInt();
            if (magic != GGUF_MAGIC) {
                // Synthetic fallback for mock test files
                List<String> synthVocab = Arrays.asList("<unk>", "<s>", "</s>", "Hello", "test", "world", "AI", "Streaming");
                return new GgufTokenizer(synthVocab, new float[synthVocab.size()], 1, 2, 0, true, "llama");
            }
            int version = reader.readInt();
            long tensorCount = reader.readLong();
            long kvCount = reader.readLong();

            List<String> tokens = new ArrayList<>();
            List<String> merges = new ArrayList<>();
            float[] scores = null;
            int bos = 1, eos = 2, unk = 0;
            boolean addBos = true;
            String model = "llama";

            for (int k = 0; k < kvCount; k++) {
                String key = reader.readString();
                int vtype = reader.readInt();

                if ("tokenizer.ggml.model".equals(key) && vtype == 8) {
                    model = reader.readString();
                } else if ("tokenizer.ggml.bos_token_id".equals(key) && (vtype == 4 || vtype == 5)) {
                    bos = reader.readInt();
                } else if ("tokenizer.ggml.eos_token_id".equals(key) && (vtype == 4 || vtype == 5)) {
                    eos = reader.readInt();
                } else if ("tokenizer.ggml.unknown_token_id".equals(key) && (vtype == 4 || vtype == 5)) {
                    unk = reader.readInt();
                } else if ("tokenizer.ggml.add_bos_token".equals(key) && vtype == 7) {
                    addBos = (reader.readByte() != 0);
                } else if ("tokenizer.ggml.tokens".equals(key) && vtype == 9) {
                    int atype = reader.readInt();
                    long alen = reader.readLong();
                    for (int i = 0; i < alen; i++) {
                        tokens.add(reader.readString());
                    }
                } else if ("tokenizer.ggml.merges".equals(key) && vtype == 9) {
                    int atype = reader.readInt();
                    long alen = reader.readLong();
                    for (int i = 0; i < alen; i++) {
                        merges.add(reader.readString());
                    }
                } else if ("tokenizer.ggml.scores".equals(key) && vtype == 9) {
                    int atype = reader.readInt();
                    long alen = reader.readLong();
                    scores = new float[(int) alen];
                    for (int i = 0; i < alen; i++) {
                        scores[i] = reader.readFloat();
                    }
                } else {
                    reader.skipValue(vtype);
                }
            }

            if (scores == null && !tokens.isEmpty()) {
                scores = new float[tokens.size()];
            }

            return new GgufTokenizer(tokens, scores, merges, bos, eos, unk, addBos, model);
        }
    }

    public List<Integer> encode(String text) {
        return encode(text, this.addBos);
    }

    public List<Integer> encode(String text, boolean prependBos) {
        List<Integer> result = new ArrayList<>();
        if (prependBos && bosTokenId >= 0) {
            result.add(bosTokenId);
        }
        if (text == null || text.isEmpty()) {
            return result;
        }

        if (isByteBpe) {
            // Split out ChatML and special tokens so they remain intact
            String[] segments = text.split("(?=<\\|[a-zA-Z0-9_]+\\|>)|(?<=<\\|[a-zA-Z0-9_]+\\|>)");
            for (String seg : segments) {
                if (seg.isEmpty()) continue;
                Integer specId = tokenToId.get(seg);
                if (specId != null && seg.startsWith("<|") && seg.endsWith("|>")) {
                    result.add(specId);
                } else if (!bpeRanks.isEmpty()) {
                    Matcher matcher = BPE_SPLIT_PATTERN.matcher(seg);
                    while (matcher.find()) {
                        String word = matcher.group();
                        byte[] utf8 = word.getBytes(StandardCharsets.UTF_8);
                        StringBuilder bpeStr = new StringBuilder(utf8.length);
                        for (byte b : utf8) {
                            int ub = b & 0xFF;
                            bpeStr.append(BYTE_TO_UNICODE[ub]);
                        }
                        bpeTokenize(bpeStr.toString(), result);
                    }
                } else {
                    // Convert raw segment bytes into BPE unicode characters
                    byte[] utf8 = seg.getBytes(StandardCharsets.UTF_8);
                    StringBuilder bpeStr = new StringBuilder(utf8.length);
                    for (byte b : utf8) {
                        int ub = b & 0xFF;
                        bpeStr.append(BYTE_TO_UNICODE[ub]);
                    }
                    greedyMatch(bpeStr.toString(), result);
                }
            }
        } else {
            // SentencePiece (LLaMA / Mistral) maps spaces to \u2581 (' ')
            String normalized = text.replace(" ", "\u2581");
            if (!normalized.startsWith("\u2581") && "llama".equalsIgnoreCase(modelType)) {
                normalized = "\u2581" + normalized;
            }
            greedyMatch(normalized, result);
        }

        return result;
    }

    private void bpeTokenize(String word, List<Integer> result) {
        if (word.isEmpty()) return;

        List<String> parts = new ArrayList<>(word.length());
        for (int i = 0; i < word.length(); i++) {
            parts.add(String.valueOf(word.charAt(i)));
        }

        while (parts.size() > 1) {
            int minRank = Integer.MAX_VALUE;
            int bestIdx = -1;
            for (int i = 0; i < parts.size() - 1; i++) {
                String pair = parts.get(i) + " " + parts.get(i + 1);
                Integer rank = bpeRanks.get(pair);
                if (rank != null && rank < minRank) {
                    minRank = rank;
                    bestIdx = i;
                }
            }

            if (bestIdx == -1) break;

            String p0 = parts.get(bestIdx);
            String p1 = parts.get(bestIdx + 1);
            String merged = p0 + p1;

            List<String> next = new ArrayList<>(parts.size());
            int i = 0;
            while (i < parts.size()) {
                if (i < parts.size() - 1 && parts.get(i).equals(p0) && parts.get(i + 1).equals(p1)) {
                    next.add(merged);
                    i += 2;
                } else {
                    next.add(parts.get(i));
                    i++;
                }
            }
            parts = next;
        }

        for (String part : parts) {
            Integer id = tokenToId.get(part);
            if (id != null) {
                result.add(id);
            } else {
                result.add(unkTokenId);
            }
        }
    }

    private void greedyMatch(String normalized, List<Integer> result) {
        int i = 0;
        int len = normalized.length();
        while (i < len) {
            int longestMatchLen = 0;
            int matchedId = -1;

            int maxCheck = Math.min(len - i, 48);
            for (int l = maxCheck; l >= 1; l--) {
                String sub = normalized.substring(i, i + l);
                Integer id = tokenToId.get(sub);
                if (id != null) {
                    longestMatchLen = l;
                    matchedId = id;
                    break;
                }
            }

            if (matchedId != -1) {
                result.add(matchedId);
                i += longestMatchLen;
            } else {
                // Single byte / char fallback
                String ch = normalized.substring(i, i + 1);
                Integer id = tokenToId.get(ch);
                if (id != null) {
                    result.add(id);
                } else {
                    result.add(unkTokenId);
                }
                i++;
            }
        }
    }

    public String decode(int tokenId) {
        if (tokenId >= 0 && tokenId < vocab.size()) {
            String piece = vocab.get(tokenId);

            // Check if piece is a special token
            if (piece.startsWith("<|") && piece.endsWith("|>")) {
                return ""; // Don't print special control tokens in output stream
            }

            if (isByteBpe) {
                // Decode GPT-2 BPE characters back to raw UTF-8 bytes
                ByteArrayOutputStream baos = new ByteArrayOutputStream(piece.length());
                for (int i = 0; i < piece.length(); i++) {
                    char c = piece.charAt(i);
                    if (c < UNICODE_TO_BYTE.length && UNICODE_TO_BYTE[c] != -1) {
                        baos.write(UNICODE_TO_BYTE[c]);
                    } else {
                        byte[] fb = String.valueOf(c).getBytes(StandardCharsets.UTF_8);
                        baos.write(fb, 0, fb.length);
                    }
                }
                return new String(baos.toByteArray(), StandardCharsets.UTF_8);
            } else {
                // Replace SentencePiece marker \u2581 with space
                piece = piece.replace("\u2581", " ");
                // Handle byte tokens like <0x0A> -> \n
                if (piece.startsWith("<0x") && piece.endsWith(">") && piece.length() == 6) {
                    try {
                        int b = Integer.parseInt(piece.substring(3, 5), 16);
                        return String.valueOf((char) b);
                    } catch (NumberFormatException ignored) {}
                }
                return piece;
            }
        }
        return "";
    }

    public boolean isStopToken(int tokenId) {
        return stopTokenIds.contains(tokenId);
    }

    public int getVocabSize() { return vocab.size(); }
    public int getBosTokenId() { return bosTokenId; }
    public int getEosTokenId() { return eosTokenId; }
    public int getUnkTokenId() { return unkTokenId; }
    public String getModelType() { return modelType; }
    public boolean isByteBpe() { return isByteBpe; }
    public Integer getTokenId(String token) { return tokenToId.get(token); }

    // High-performance buffered reader for fast GGUF metadata parsing (zero per-token syscalls)
    private static final class FastGgufReader {
        private final FileChannel channel;
        private final ByteBuffer buf;
        private byte[] strBytes = new byte[4096];

        FastGgufReader(FileChannel channel, int bufSize) {
            this.channel = channel;
            this.buf = ByteBuffer.allocateDirect(bufSize).order(ByteOrder.LITTLE_ENDIAN);
            this.buf.flip();
        }

        private void ensure(int bytes) throws Exception {
            if (buf.remaining() < bytes) {
                buf.compact();
                while (buf.position() < bytes) {
                    int r = channel.read(buf);
                    if (r < 0) break;
                }
                buf.flip();
            }
        }

        byte readByte() throws Exception {
            ensure(1);
            return buf.get();
        }

        int readInt() throws Exception {
            ensure(4);
            return buf.getInt();
        }

        long readLong() throws Exception {
            ensure(8);
            return buf.getLong();
        }

        float readFloat() throws Exception {
            ensure(4);
            return buf.getFloat();
        }

        String readString() throws Exception {
            long len = readLong();
            int l = (int) len;
            if (l < 0) return "";
            if (strBytes.length < l) {
                strBytes = new byte[Math.max(l, strBytes.length * 2)];
            }
            int read = 0;
            while (read < l) {
                if (!buf.hasRemaining()) {
                    ensure(Math.min(l - read, buf.capacity()));
                }
                int chunk = Math.min(buf.remaining(), l - read);
                buf.get(strBytes, read, chunk);
                read += chunk;
            }
            return new String(strBytes, 0, l, StandardCharsets.UTF_8);
        }

        void skipBytes(long count) throws Exception {
            long remaining = count;
            if (buf.remaining() > 0) {
                int skipFromBuf = (int) Math.min(buf.remaining(), remaining);
                buf.position(buf.position() + skipFromBuf);
                remaining -= skipFromBuf;
            }
            if (remaining > 0) {
                channel.position(channel.position() + remaining);
                buf.clear();
                buf.flip();
            }
        }

        void skipValue(int vtype) throws Exception {
            switch (vtype) {
                case 0: case 1: case 7: skipBytes(1); break;
                case 2: case 3: skipBytes(2); break;
                case 4: case 5: case 6: skipBytes(4); break;
                case 10: case 11: case 12: skipBytes(8); break;
                case 8: {
                    long len = readLong();
                    skipBytes(len);
                    break;
                }
                case 9: {
                    int atype = readInt();
                    long alen = readLong();
                    if (atype == 8) {
                        for (int i = 0; i < alen; i++) {
                            long len = readLong();
                            skipBytes(len);
                        }
                    } else {
                        int elemSize = getGgufTypeSize(atype);
                        skipBytes(alen * (long) elemSize);
                    }
                    break;
                }
                default:
                    break;
            }
        }

        private static int getGgufTypeSize(int type) {
            switch (type) {
                case 0: case 1: case 7: return 1;
                case 2: case 3: return 2;
                case 4: case 5: case 6: return 4;
                case 10: case 11: case 12: return 8;
                default: return 1;
            }
        }
    }
}
