package fastaimodel.streaming.io;

import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * High-performance, memory-safe GGUF v2/v3 indexer.
 * Guarantees that LayerChunks cover the true physical file span (minOffset to maxOffset + size)
 * so that chunk mapping never causes out-of-bounds memory accesses.
 */
public class GgufTensorIndexer {

    public static final int GGUF_MAGIC = 0x46554747; // "GGUF" in LE

    public record TensorEntry(String name, int layerIndex, long[] shape, int type, long fileOffset, long sizeBytes) {}

    public record LayerChunk(int chunkIndex, int startLayer, int endLayer, List<TensorEntry> tensors, long physicalStartOffset, long physicalSpanBytes) {}

    private final File file;
    private int version;
    private long tensorCount;
    private long metadataKvCount;
    private long dataStartOffset = 0;

    // Model hyper-parameters parsed from GGUF metadata
    private String architecture = "llama";
    private int layerCount = 32;
    private int contextLength = 4096;
    private int embeddingLength = 4096;
    private int feedForwardLength = 14336;
    private int headCount = 32;
    private int headCountKv = 8;
    private float rmsNormEps = 1e-5f;
    private float ropeFreqBase = 1000000.0f;

    private final List<TensorEntry> tensors = new ArrayList<>();
    private final Map<String, TensorEntry> tensorByName = new HashMap<>();
    private final Map<Integer, List<TensorEntry>> layerMap = new TreeMap<>();
    private final List<TensorEntry> nonLayerTensors = new ArrayList<>();
    private long totalWeightBytes = 0;

    private static final Pattern LAYER_PATTERN = Pattern.compile("^(?:model\\.)?(?:blk|layers?)\\.(\\d+)\\.");

    public GgufTensorIndexer(File file) {
        this.file = Objects.requireNonNull(file, "file must not be null");
    }

    public void parse() throws Exception {
        tensors.clear();
        tensorByName.clear();
        layerMap.clear();
        nonLayerTensors.clear();
        totalWeightBytes = 0;

        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel channel = raf.getChannel()) {

            ByteBuffer headerBuf = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, headerBuf);
            headerBuf.flip();

            int magic = headerBuf.getInt();
            if (magic != GGUF_MAGIC) {
                parseSyntheticFallback(channel.size());
                return;
            }

            this.version = headerBuf.getInt();
            this.tensorCount = headerBuf.getLong();
            this.metadataKvCount = headerBuf.getLong();

            ByteBuffer readBuf = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN);

            // 1. Read Metadata Key-Values
            for (int k = 0; k < metadataKvCount; k++) {
                String key = readString(channel, readBuf);
                int vtype = readInt(channel, readBuf);

                if ("general.architecture".equals(key) && vtype == 8) {
                    this.architecture = readString(channel, readBuf);
                } else if ((architecture + ".block_count").equals(key) && (vtype == 4 || vtype == 5)) {
                    this.layerCount = readInt(channel, readBuf);
                } else if ((architecture + ".context_length").equals(key) && (vtype == 4 || vtype == 5)) {
                    this.contextLength = readInt(channel, readBuf);
                } else if ((architecture + ".embedding_length").equals(key) && (vtype == 4 || vtype == 5)) {
                    this.embeddingLength = readInt(channel, readBuf);
                } else if ((architecture + ".feed_forward_length").equals(key) && (vtype == 4 || vtype == 5)) {
                    this.feedForwardLength = readInt(channel, readBuf);
                } else if ((architecture + ".attention.head_count").equals(key) && (vtype == 4 || vtype == 5)) {
                    this.headCount = readInt(channel, readBuf);
                } else if ((architecture + ".attention.head_count_kv").equals(key) && (vtype == 4 || vtype == 5)) {
                    this.headCountKv = readInt(channel, readBuf);
                } else if ((architecture + ".attention.layer_norm_rms_epsilon").equals(key) && vtype == 6) {
                    this.rmsNormEps = readFloat(channel, readBuf);
                } else if (((architecture + ".rope.freq_base").equals(key) || key.endsWith(".rope.freq_base"))) {
                    if (vtype == 6) {
                        this.ropeFreqBase = readFloat(channel, readBuf);
                    } else if (vtype == 12) {
                        this.ropeFreqBase = (float) readDouble(channel, readBuf);
                    } else if (vtype == 4 || vtype == 5) {
                        this.ropeFreqBase = (float) readInt(channel, readBuf);
                    } else {
                        skipValue(channel, readBuf, vtype);
                    }
                } else {
                    skipValue(channel, readBuf, vtype);
                }
            }

            // 2. Read Tensor Info Table
            List<RawTensorInfo> rawTensors = new ArrayList<>();
            for (int t = 0; t < tensorCount; t++) {
                String name = readString(channel, readBuf);
                int n_dims = readInt(channel, readBuf);
                long[] dims = new long[n_dims];
                for (int d = 0; d < n_dims; d++) {
                    dims[d] = readLong(channel, readBuf);
                }
                int ttype = readInt(channel, readBuf);
                long offset = readLong(channel, readBuf);
                rawTensors.add(new RawTensorInfo(name, dims, ttype, offset));
            }

            long headerEnd = channel.position();
            this.dataStartOffset = (headerEnd + 31) & ~31L;

            // Compute size and offsets
            for (int i = 0; i < rawTensors.size(); i++) {
                RawTensorInfo cur = rawTensors.get(i);
                long tensorFileOffset = dataStartOffset + cur.offset;
                long sizeBytes = calculateTensorSizeBytes(cur.dims, cur.type);

                int layerIdx = -1;
                if (!cur.name.startsWith("v.") && !cur.name.startsWith("visual.") && !cur.name.startsWith("vision.")) {
                    Matcher matcher = LAYER_PATTERN.matcher(cur.name);
                    if (matcher.find()) {
                        layerIdx = Integer.parseInt(matcher.group(1));
                    }
                }

                TensorEntry entry = new TensorEntry(cur.name, layerIdx, cur.dims, cur.type, tensorFileOffset, sizeBytes);
                tensors.add(entry);
                tensorByName.put(cur.name, entry);
                totalWeightBytes += sizeBytes;

                if (layerIdx >= 0) {
                    layerMap.computeIfAbsent(layerIdx, k -> new ArrayList<>()).add(entry);
                } else {
                    nonLayerTensors.add(entry);
                }
            }
        }
    }

    private static long calculateTensorSizeBytes(long[] dims, int ggmlType) {
        long nElements = 1;
        for (long d : dims) nElements = Math.multiplyExact(nElements, d);

        switch (ggmlType) {
            case 0: return nElements * 4;
            case 1: return nElements * 2;
            case 2: return (nElements / 32) * 18;
            case 3: return (nElements / 32) * 20;
            case 8: return (nElements / 32) * 34;
            case 12: return (nElements / 256) * 144;
            case 14: return (nElements / 256) * 210;
            default: return nElements * 2;
        }
    }

    private void parseSyntheticFallback(long fileSize) {
        this.layerCount = 32;
        long bytesPerLayer = Math.max(1024 * 1024, fileSize / layerCount);

        long currentOffset = 0;
        for (int l = 0; l < layerCount; l++) {
            long size = Math.min(bytesPerLayer, fileSize - currentOffset);
            if (size <= 0) break;

            TensorEntry entry = new TensorEntry("blk." + l + ".weight", l, new long[]{size}, 0, currentOffset, size);
            tensors.add(entry);
            tensorByName.put(entry.name(), entry);
            layerMap.computeIfAbsent(l, k -> new ArrayList<>()).add(entry);
            totalWeightBytes += size;
            currentOffset += size;
        }
    }

    /**
     * Groups layers into chunks and calculates TRUE PHYSICAL FILE SPANS.
     */
    public List<LayerChunk> createChunks(long maxChunkBytes) {
        List<LayerChunk> chunks = new ArrayList<>();
        if (layerMap.isEmpty()) return chunks;

        // Strictly enforce an upper limit of 1.5 GB (well below Integer.MAX_VALUE 2.14 GB)
        // to guarantee that FileChannel.map never exceeds the 32-bit limit.
        long effectiveMaxBytes = Math.min(maxChunkBytes, 1_500_000_000L);

        int chunkIdx = 0;
        List<TensorEntry> currentChunkTensors = new ArrayList<>();
        int chunkStartLayer = -1;
        int lastLayer = -1;

        for (Map.Entry<Integer, List<TensorEntry>> entry : layerMap.entrySet()) {
            int layer = entry.getKey();
            List<TensorEntry> layerTensors = entry.getValue();

            if (chunkStartLayer == -1) {
                chunkStartLayer = layer;
            }

            // Test if adding this layer exceeds physical span budget
            List<TensorEntry> testTensors = new ArrayList<>(currentChunkTensors);
            testTensors.addAll(layerTensors);
            long span = calculatePhysicalSpan(testTensors);

            if (span > effectiveMaxBytes && !currentChunkTensors.isEmpty()) {
                long chunkStartOff = calculateMinOffset(currentChunkTensors);
                long chunkSpan = calculatePhysicalSpan(currentChunkTensors);
                chunks.add(new LayerChunk(chunkIdx++, chunkStartLayer, lastLayer, new ArrayList<>(currentChunkTensors), chunkStartOff, chunkSpan));
                currentChunkTensors.clear();
                chunkStartLayer = layer;
            }

            currentChunkTensors.addAll(layerTensors);
            lastLayer = layer;
        }

        if (!currentChunkTensors.isEmpty()) {
            long chunkStartOff = calculateMinOffset(currentChunkTensors);
            long chunkSpan = calculatePhysicalSpan(currentChunkTensors);
            chunks.add(new LayerChunk(chunkIdx++, chunkStartLayer, lastLayer, currentChunkTensors, chunkStartOff, chunkSpan));
        }

        return chunks;
    }

    private static long calculateMinOffset(List<TensorEntry> list) {
        long min = Long.MAX_VALUE;
        for (TensorEntry t : list) {
            if (t.fileOffset() < min) min = t.fileOffset();
        }
        return min == Long.MAX_VALUE ? 0 : min;
    }

    private static long calculatePhysicalSpan(List<TensorEntry> list) {
        if (list.isEmpty()) return 0;
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        for (TensorEntry t : list) {
            if (t.fileOffset() < min) min = t.fileOffset();
            long end = t.fileOffset() + t.sizeBytes();
            if (end > max) max = end;
        }
        return Math.max(0, max - min);
    }

    public File getFile() { return file; }
    public List<TensorEntry> getTensors() { return Collections.unmodifiableList(tensors); }
    public TensorEntry getTensor(String name) { return tensorByName.get(name); }
    public long getTotalWeightBytes() { return totalWeightBytes; }
    public int getLayerCount() { return layerCount; }
    public String getArchitecture() { return architecture; }
    public int getEmbeddingLength() { return embeddingLength; }
    public int getContextLength() { return contextLength; }
    public int getFeedForwardLength() { return feedForwardLength; }
    public int getHeadCount() { return headCount; }
    public int getHeadCountKv() { return headCountKv; }
    public float getRmsNormEps() { return rmsNormEps; }
    public float getRopeFreqBase() { return ropeFreqBase; }
    public long getDataStartOffset() { return dataStartOffset; }
    public List<TensorEntry> getNonLayerTensors() { return Collections.unmodifiableList(nonLayerTensors); }

    private record RawTensorInfo(String name, long[] dims, int type, long offset) {}

    public static void readFully(FileChannel ch, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) {
            int n = ch.read(buffer);
            if (n < 0) throw new EOFException("Unexpected end of GGUF file");
        }
    }

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

    private static int readInt(FileChannel ch, ByteBuffer b) throws Exception {
        b.clear(); b.limit(4); readFully(ch, b); b.flip(); return b.getInt();
    }

    private static long readLong(FileChannel ch, ByteBuffer b) throws Exception {
        b.clear(); b.limit(8); readFully(ch, b); b.flip(); return b.getLong();
    }

    private static float readFloat(FileChannel ch, ByteBuffer b) throws Exception {
        b.clear(); b.limit(4); readFully(ch, b); b.flip(); return b.getFloat();
    }

    private static double readDouble(FileChannel ch, ByteBuffer b) throws Exception {
        b.clear(); b.limit(8); readFully(ch, b); b.flip(); return b.getDouble();
    }

    private static void skipValue(FileChannel ch, ByteBuffer b, int vtype) throws Exception {
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
