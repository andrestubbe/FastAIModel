package fastaimodel.streaming.io;

import fastpointer.Pointer;
import sun.misc.Unsafe;

import java.io.File;
import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Zero-copy memory mapped file slice provider delivering raw FastPointer native addresses.
 */
public class NativeChunkMmap implements AutoCloseable {

    private static final Unsafe UNSAFE;

    static {
        Unsafe unsafe = null;
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            unsafe = (Unsafe) f.get(null);
        } catch (Throwable ignored) {}
        UNSAFE = unsafe;
    }

    private final File file;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final List<MappedByteBuffer> mappedBuffers = new ArrayList<>();

    public NativeChunkMmap(File file) throws Exception {
        this.file = Objects.requireNonNull(file, "file must not be null");
        this.raf = new RandomAccessFile(file, "r");
        this.channel = raf.getChannel();
    }

    private final java.util.Map<Long, MappedByteBuffer> pointerToBuffer = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Maps a chunk slice and returns a FastPointer to the 64-bit native virtual memory address.
     */
    public synchronized Pointer mapChunkPointer(long offset, long sizeBytes) throws Exception {
        MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, offset, sizeBytes);
        long address = getDirectBufferAddress(buffer);
        pointerToBuffer.put(address, buffer);
        mappedBuffers.add(buffer);
        return Pointer.of(address);
    }

    public synchronized void unmapPointer(Pointer pointer) {
        if (pointer == null || pointer.isNull()) return;
        MappedByteBuffer buffer = pointerToBuffer.remove(pointer.address());
        if (buffer != null) {
            mappedBuffers.remove(buffer);
            unmap(buffer);
        }
    }

    public synchronized ByteBuffer mapChunk(long offset, long sizeBytes) throws Exception {
        MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, offset, sizeBytes);
        mappedBuffers.add(buffer);
        return buffer;
    }

    public static long getDirectBufferAddress(ByteBuffer buffer) {
        if (buffer == null || !buffer.isDirect()) return 0L;
        try {
            Field addressField = java.nio.Buffer.class.getDeclaredField("address");
            addressField.setAccessible(true);
            return addressField.getLong(buffer);
        } catch (Throwable t) {
            return 0L;
        }
    }

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
            Class<?> unsafeClass = Class.forName("sun.misc.Unsafe");
            Field f = unsafeClass.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            Object unsafe = f.get(null);
            Method invokeCleaner = unsafeClass.getMethod("invokeCleaner", ByteBuffer.class);
            invokeCleaner.invoke(unsafe, buffer);
        } catch (Throwable ignored) {}
    }

    public File getFile() { return file; }
    public long getFileSize() throws Exception { return channel.size(); }
}
