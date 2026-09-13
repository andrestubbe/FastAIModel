package fastaimodel.streaming.tokenizer;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Pure Java BPE & SentencePiece Tokenizer parsing directly from GGUF metadata.
 * Supports LLaMA / Mistral SentencePiece tokenization with ' ' (\u2581) word boundaries
 * as well as GPT2/SmolLM Byte-Pair Encoding.
 */
public class GgufTokenizer {

    public static final int GGUF_MAGIC = 0x46554747;

    private final List<String> vocab = new ArrayList<>();
    private final Map<String, Integer> tokenToId = new HashMap<>();
    private final float[] scores;
    private int bosTokenId = 1;
    private int eosTokenId = 2;
    private int unkTokenId = 0;
    private boolean addBos = true;
    private boolean addEos = false;
    private String modelType = "llama";

    public GgufTokenizer(List<String> vocab, float[] scores, int bos, int eos, int unk, boolean addBos, String modelType) {
        this.vocab.addAll(vocab);
        this.scores = scores;
        this.bosTokenId = bos;
        this.eosTokenId = eos;
        this.unkTokenId = unk;
        this.addBos = addBos;
        this.modelType = modelType;

        for (int i = 0; i < vocab.size(); i++) {
            tokenToId.putIfAbsent(vocab.get(i), i);
        }
    }

    public static GgufTokenizer loadFromGguf(File ggufFile) throws Exception {
        try (RandomAccessFile raf = new RandomAccessFile(ggufFile, "r");
             FileChannel channel = raf.getChannel()) {

            ByteBuffer buf = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(buf);
            buf.flip();

            int magic = buf.getInt();
            int version = buf.getInt();
            long tensorCount = buf.getLong();
            long kvCount = buf.getLong();

            if (magic != GGUF_MAGIC) {
                throw new IllegalArgumentException("Invalid GGUF magic: " + Integer.toHexString(magic));
            }

            List<String> tokens = new ArrayList<>();
            float[] scores = null;
            int bos = 1, eos = 2, unk = 0;
            boolean addBos = true;
            String model = "llama";

            // Buffer for metadata scanning
            ByteBuffer readBuf = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN);

            for (int k = 0; k < kvCount; k++) {
                String key = readString(channel, readBuf);
                int vtype = readInt(channel, readBuf);

                if ("tokenizer.ggml.model".equals(key) && vtype == 8) {
                    model = readString(channel, readBuf);
                } else if ("tokenizer.ggml.bos_token_id".equals(key) && (vtype == 4 || vtype == 5)) {
                    bos = readInt(channel, readBuf);
                } else if ("tokenizer.ggml.eos_token_id".equals(key) && (vtype == 4 || vtype == 5)) {
                    eos = readInt(channel, readBuf);
                } else if ("tokenizer.ggml.unknown_token_id".equals(key) && (vtype == 4 || vtype == 5)) {
                    unk = readInt(channel, readBuf);
                } else if ("tokenizer.ggml.add_bos_token".equals(key) && vtype == 7) {
                    addBos = (readByte(channel, readBuf) != 0);
                } else if ("tokenizer.ggml.tokens".equals(key) && vtype == 9) {
                    int atype = readInt(channel, readBuf);
                    long alen = readLong(channel, readBuf);
                    for (int i = 0; i < alen; i++) {
                        tokens.add(readString(channel, readBuf));
                    }
                } else if ("tokenizer.ggml.scores".equals(key) && vtype == 9) {
                    int atype = readInt(channel, readBuf);
                    long alen = readLong(channel, readBuf);
                    scores = new float[(int) alen];
                    for (int i = 0; i < alen; i++) {
                        scores[i] = readFloat(channel, readBuf);
                    }
                } else {
                    skipValue(channel, readBuf, vtype);
                }
            }

            if (scores == null && !tokens.isEmpty()) {
                scores = new float[tokens.size()];
            }

            return new GgufTokenizer(tokens, scores, bos, eos, unk, addBos, model);
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

        // SentencePiece standard pre-tokenization: replace space with   (\u2581)
        String normalized = text.replace(" ", "▁");
        if (!normalized.startsWith("▁") && "llama".equalsIgnoreCase(modelType)) {
            normalized = "▁" + normalized;
        }

        // Greedy longest match / SentencePiece piece matching
        int i = 0;
        int len = normalized.length();
        while (i < len) {
            int longestMatchLen = 0;
            int matchedId = -1;

            int maxCheck = Math.min(len - i, 32);
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

        return result;
    }

    public String decode(int tokenId) {
        if (tokenId >= 0 && tokenId < vocab.size()) {
            String piece = vocab.get(tokenId);
            // Replace SentencePiece marker \u2581 with space
            piece = piece.replace("▁", " ");
            // Handle byte tokens like <0x0A> -> \n
            if (piece.startsWith("<0x") && piece.endsWith(">") && piece.length() == 6) {
                try {
                    int b = Integer.parseInt(piece.substring(3, 5), 16);
                    return String.valueOf((char) b);
                } catch (NumberFormatException ignored) {}
            }
            return piece;
        }
        return "";
    }

    public int getVocabSize() { return vocab.size(); }
    public int getBosTokenId() { return bosTokenId; }
    public int getEosTokenId() { return eosTokenId; }
    public int getUnkTokenId() { return unkTokenId; }
    public String getModelType() { return modelType; }

    // Helpers for binary reading
    private static String readString(FileChannel ch, ByteBuffer b) throws Exception {
        long len = readLong(ch, b);
        byte[] bytes = new byte[(int) len];
        int read = 0;
        while (read < len) {
            b.clear();
            b.limit((int) Math.min(b.capacity(), len - read));
            int r = ch.read(b);
            if (r < 0) break;
            b.flip();
            b.get(bytes, read, r);
            read += r;
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static byte readByte(FileChannel ch, ByteBuffer b) throws Exception {
        b.clear(); b.limit(1); ch.read(b); b.flip(); return b.get();
    }

    private static int readInt(FileChannel ch, ByteBuffer b) throws Exception {
        b.clear(); b.limit(4); ch.read(b); b.flip(); return b.getInt();
    }

    private static long readLong(FileChannel ch, ByteBuffer b) throws Exception {
        b.clear(); b.limit(8); ch.read(b); b.flip(); return b.getLong();
    }

    private static float readFloat(FileChannel ch, ByteBuffer b) throws Exception {
        b.clear(); b.limit(4); ch.read(b); b.flip(); return b.getFloat();
    }

    private static void skipValue(FileChannel ch, ByteBuffer b, int vtype) throws Exception {
        // vtypes: 0=u8, 1=i8, 2=u16, 3=i16, 4=u32, 5=i32, 6=f32, 7=bool, 8=str, 9=arr, 10=u64, 11=i64, 12=f64
        switch (vtype) {
            case 0: case 1: case 7: ch.position(ch.position() + 1); break;
            case 2: case 3: ch.position(ch.position() + 2); break;
            case 4: case 5: case 6: ch.position(ch.position() + 4); break;
            case 10: case 11: case 12: ch.position(ch.position() + 8); break;
            case 8: {
                long len = readLong(ch, b);
                ch.position(ch.position() + len);
                break;
            }
            case 9: {
                int atype = readInt(ch, b);
                long alen = readLong(ch, b);
                if (atype == 8) {
                    for (int i = 0; i < alen; i++) {
                        long sl = readLong(ch, b);
                        ch.position(ch.position() + sl);
                    }
                } else if (atype == 4 || atype == 5 || atype == 6) {
                    ch.position(ch.position() + alen * 4);
                } else if (atype == 10 || atype == 11 || atype == 12) {
                    ch.position(ch.position() + alen * 8);
                } else if (atype == 0 || atype == 1 || atype == 7) {
                    ch.position(ch.position() + alen);
                } else if (atype == 2 || atype == 3) {
                    ch.position(ch.position() + alen * 2);
                }
                break;
            }
            default:
                break;
        }
    }
}
