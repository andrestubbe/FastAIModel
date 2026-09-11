package fastaimodel.streaming.buffer;

import fastmemory.Memory;
import fastpointer.Pointer;

/**
 * Dual-slot off-heap ring buffer (Slot A & Slot B) backed by 32-byte SIMD-aligned FastMemory.
 * Eliminates Java 2 GB buffer limits by operating on 64-bit native memory pointers.
 */
public class DoubleBufferRing implements AutoCloseable {

    public enum SlotState { EMPTY, LOADING, READY, IN_USE }

    public static class MemorySlot {
        private final int slotId;
        private final Memory memory;
        private final Pointer pointer;
        private final long capacityBytes;
        private volatile SlotState state = SlotState.EMPTY;
        private volatile int currentChunkIndex = -1;

        public MemorySlot(int slotId, long capacityBytes) {
            this.slotId = slotId;
            this.capacityBytes = capacityBytes;
            // Allocate 32-byte SIMD-aligned native memory
            this.memory = Memory.allocateAligned(capacityBytes, 32);
            this.pointer = memory.pointer();
            // Lock physical RAM pages to prevent OS page faults during inference
            this.memory.lockPages();
        }

        public int getSlotId() { return slotId; }
        public Memory getMemory() { return memory; }
        public Pointer getPointer() { return pointer; }
        public long getCapacityBytes() { return capacityBytes; }
        public SlotState getState() { return state; }
        public void setState(SlotState state) { this.state = state; }
        public int getCurrentChunkIndex() { return currentChunkIndex; }
        public void setCurrentChunkIndex(int index) { this.currentChunkIndex = index; }

        public void close() {
            memory.close();
        }
    }

    private final MemorySlot slotA;
    private final MemorySlot slotB;
    private volatile int activeComputeSlotId = 0; // 0 for A, 1 for B
    private final long slotCapacityBytes;

    public DoubleBufferRing(long slotCapacityBytes) {
        this.slotCapacityBytes = slotCapacityBytes;
        this.slotA = new MemorySlot(0, slotCapacityBytes);
        this.slotB = new MemorySlot(1, slotCapacityBytes);
    }

    public MemorySlot getActiveComputeSlot() {
        return activeComputeSlotId == 0 ? slotA : slotB;
    }

    public MemorySlot getPrefetchSlot() {
        return activeComputeSlotId == 0 ? slotB : slotA;
    }

    /**
     * Sub-microsecond atomic slot pointer swap.
     */
    public synchronized void swapSlots() {
        activeComputeSlotId = (activeComputeSlotId == 0) ? 1 : 0;
    }

    public long getSlotCapacityBytes() { return slotCapacityBytes; }
    public MemorySlot getSlotA() { return slotA; }
    public MemorySlot getSlotB() { return slotB; }

    @Override
    public void close() {
        slotA.close();
        slotB.close();
    }
}
