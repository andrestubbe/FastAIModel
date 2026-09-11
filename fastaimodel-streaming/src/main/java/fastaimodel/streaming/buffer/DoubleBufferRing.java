package fastaimodel.streaming.buffer;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * Dual-slot off-heap ring buffer (Slot A & Slot B).
 * Enforces strict memory caps by recycling fixed pre-allocated direct buffers.
 */
public class DoubleBufferRing {

    public enum SlotState { EMPTY, LOADING, READY, IN_USE }

    public static class MemorySlot {
        private final int slotId;
        private final ByteBuffer buffer;
        private volatile SlotState state = SlotState.EMPTY;
        private volatile int currentChunkIndex = -1;

        public MemorySlot(int slotId, int capacityBytes) {
            this.slotId = slotId;
            this.buffer = ByteBuffer.allocateDirect(capacityBytes);
        }

        public int getSlotId() { return slotId; }
        public ByteBuffer getBuffer() { return buffer; }
        public SlotState getState() { return state; }
        public void setState(SlotState state) { this.state = state; }
        public int getCurrentChunkIndex() { return currentChunkIndex; }
        public void setCurrentChunkIndex(int index) { this.currentChunkIndex = index; }
    }

    private final MemorySlot slotA;
    private final MemorySlot slotB;
    private volatile int activeComputeSlotId = 0; // 0 for A, 1 for B
    private final int slotCapacityBytes;

    public DoubleBufferRing(int slotCapacityBytes) {
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
     * Swaps compute and prefetch slots in sub-microsecond time.
     */
    public synchronized void swapSlots() {
        activeComputeSlotId = (activeComputeSlotId == 0) ? 1 : 0;
    }

    public int getSlotCapacityBytes() { return slotCapacityBytes; }
    public MemorySlot getSlotA() { return slotA; }
    public MemorySlot getSlotB() { return slotB; }
}
