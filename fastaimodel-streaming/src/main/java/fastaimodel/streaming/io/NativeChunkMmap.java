package fastaimodel.streaming.io;

import java.io.File;
import java.io.RandomAccessFile;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Zero-copy memory mapped file slice provider with explicit Windows buffer unmapping.
 */
public class NativeChunkMmap implements AutoCloseable {

    private final File file;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final List<MappedByteBuffer> mappedBuffers = new ArrayList<>();

    public NativeChunkMmap(File file) throws Exception {
        this.file = Objects.requireNonNull(file, "file must not be null");
        this.raf = new RandomAccessFile(file, "r");
        this.channel = raf.getChannel();
    }

    /**
     * Maps a chunk slice into direct off-heap virtual memory.
     */
    public synchronized ByteBuffer mapChunk(long offset, long sizeBytes) throws Exception {
        MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, offset, sizeBytes);
        mappedBuffers.add(buffer);
        return buffer;
    }

    /**
     * Unmaps all mapped direct buffers and closes the underlying file handle.
     */
    @Override
    public synchronized void close() throws Exception {
        for (MappedByteBuffer buffer : mappedBuffers) {
            unmap(buffer);
        }
        mappedBuffers.clear();

        if (channel != null && channel.isOpen()) {
            channel.close();
        }
        if (raf != null) {
            raf.close();
        }
    }

    private static void unmap(MappedByteBuffer buffer) {
        if (buffer == null) return;
        try {
            // Java 9+ unsafe unmap via cleaner
            Class<?> unsafeClass = Class.forName("sun.misc.Unsafe");
            java.lang.reflect.Field f = unsafeClass.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            Object unsafe = f.get(null);
            Method invokeCleaner = unsafeClass.getMethod("invokeCleaner", ByteBuffer.class);
            invokeCleaner.invoke(unsafe, buffer);
        } catch (Throwable ignored) {
            // Fallback for older JVMs or restricted access
        }
    }

    public File getFile() { return file; }
    public long getFileSize() throws Exception { return channel.size(); }
}
