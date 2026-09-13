package fastaimodel.streaming.compute;

import fastpointer.Pointer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * High-performance pure-Java & SIMD-compatible dequantization kernels for GGUF.
 * Supports F32 (type 0), F16 (type 1), Q4_0 (type 2), Q8_0 (type 8), Q4_K (type 12), and Q6_K (type 14).
 */
public final class GgufDequantizer {

    private GgufDequantizer() {}

    /**
     * Dequantize weight bytes into float array output.
     */
    public static void dequantize(int type, byte[] src, int srcOffset, float[] dst, int dstOffset, int count) {
        switch (type) {
            case 0: // F32
                dequantizeF32(src, srcOffset, dst, dstOffset, count);
                break;
            case 1: // F16
                dequantizeF16(src, srcOffset, dst, dstOffset, count);
                break;
            case 2: // Q4_0
                dequantizeQ4_0(src, srcOffset, dst, dstOffset, count);
                break;
            case 8: // Q8_0
                dequantizeQ8_0(src, srcOffset, dst, dstOffset, count);
                break;
            case 12: // Q4_K
                dequantizeQ4_K(src, srcOffset, dst, dstOffset, count);
                break;
            case 14: // Q6_K
                dequantizeQ6_K(src, srcOffset, dst, dstOffset, count);
                break;
            default:
                // Fallback zero fill
                for (int i = 0; i < count; i++) dst[dstOffset + i] = 0.0f;
                break;
        }
    }

    public static void dequantizeF32(byte[] src, int srcOffset, float[] dst, int dstOffset, int count) {
        ByteBuffer buf = ByteBuffer.wrap(src, srcOffset, count * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            dst[dstOffset + i] = buf.getFloat();
        }
    }

    public static void dequantizeF16(byte[] src, int srcOffset, float[] dst, int dstOffset, int count) {
        ByteBuffer buf = ByteBuffer.wrap(src, srcOffset, count * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            dst[dstOffset + i] = halfToFloat(buf.getShort());
        }
    }

    public static void dequantizeQ4_0(byte[] src, int srcOffset, float[] dst, int dstOffset, int count) {
        // Block size = 32. Each block has 2 bytes f16 delta + 16 bytes quantized quants (32 4-bit nibbles)
        int nb = count / 32;
        int idx = srcOffset;
        int outIdx = dstOffset;

        for (int b = 0; b < nb; b++) {
            short dRaw = (short) ((src[idx] & 0xFF) | ((src[idx + 1] & 0xFF) << 8));
            float d = halfToFloat(dRaw);
            idx += 2;

            for (int i = 0; i < 16; i++) {
                int val = src[idx + i] & 0xFF;
                int v0 = (val & 0x0F) - 8;
                int v1 = (val >> 4) - 8;
                dst[outIdx + i] = v0 * d;
                dst[outIdx + i + 16] = v1 * d;
            }
            idx += 16;
            outIdx += 32;
        }
    }

    public static void dequantizeQ8_0(byte[] src, int srcOffset, float[] dst, int dstOffset, int count) {
        // Block size = 32. 2 bytes f16 delta + 32 bytes int8
        int nb = count / 32;
        int idx = srcOffset;
        int outIdx = dstOffset;

        for (int b = 0; b < nb; b++) {
            short dRaw = (short) ((src[idx] & 0xFF) | ((src[idx + 1] & 0xFF) << 8));
            float d = halfToFloat(dRaw);
            idx += 2;

            for (int i = 0; i < 32; i++) {
                dst[outIdx + i] = src[idx + i] * d;
            }
            idx += 32;
            outIdx += 32;
        }
    }

    public static void dequantizeQ4_K(byte[] src, int srcOffset, float[] dst, int dstOffset, int count) {
        // Q4_K super-block size = 256 elements, 144 bytes per block.
        // Matches ggml-quants.c dequantize_row_q4_K exactly.
        int nb = count / 256;
        int idx = srcOffset;
        int outIdx = dstOffset;

        for (int i = 0; i < nb; i++) {
            short dRaw = (short) ((src[idx] & 0xFF) | ((src[idx + 1] & 0xFF) << 8));
            short dminRaw = (short) ((src[idx + 2] & 0xFF) | ((src[idx + 3] & 0xFF) << 8));
            float d = halfToFloat(dRaw);
            float min = halfToFloat(dminRaw);

            int scalesIdx = idx + 4;
            int qsIdx = idx + 16;

            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                // get_scale_min_k4(is + 0)
                int sc1, m1;
                int j0 = is;
                if (j0 < 4) {
                    sc1 = src[scalesIdx + j0] & 63;
                    m1  = src[scalesIdx + j0 + 4] & 63;
                } else {
                    sc1 = (src[scalesIdx + j0 + 4] & 0xF) | (((src[scalesIdx + j0 - 4] & 0xFF) >> 6) << 4);
                    m1  = ((src[scalesIdx + j0 + 4] & 0xFF) >> 4) | (((src[scalesIdx + j0] & 0xFF) >> 6) << 4);
                }
                float d1 = d * sc1;
                float m1f = min * m1;

                // get_scale_min_k4(is + 1)
                int sc2, m2;
                int j1 = is + 1;
                if (j1 < 4) {
                    sc2 = src[scalesIdx + j1] & 63;
                    m2  = src[scalesIdx + j1 + 4] & 63;
                } else {
                    sc2 = (src[scalesIdx + j1 + 4] & 0xF) | (((src[scalesIdx + j1 - 4] & 0xFF) >> 6) << 4);
                    m2  = ((src[scalesIdx + j1 + 4] & 0xFF) >> 4) | (((src[scalesIdx + j1] & 0xFF) >> 6) << 4);
                }
                float d2 = d * sc2;
                float m2f = min * m2;

                for (int l = 0; l < 32; ++l) {
                    int q = src[qsIdx + l] & 0xFF;
                    dst[outIdx++] = d1 * (q & 0xF) - m1f;
                }
                for (int l = 0; l < 32; ++l) {
                    int q = src[qsIdx + l] & 0xFF;
                    dst[outIdx++] = d2 * (q >> 4) - m2f;
                }

                qsIdx += 32;
                is += 2;
            }

            idx += 144;
        }
    }

    public static void dequantizeQ6_K(byte[] src, int srcOffset, float[] dst, int dstOffset, int count) {
        // Q6_K super-block size = 256 elements, 210 bytes per block.
        // Matches ggml-quants.c dequantize_row_q6_K exactly.
        int nb = count / 256;
        int idx = srcOffset;
        int outIdx = dstOffset;

        for (int i = 0; i < nb; i++) {
            short dRaw = (short) ((src[idx + 208] & 0xFF) | ((src[idx + 209] & 0xFF) << 8));
            float d = halfToFloat(dRaw);

            int qlIdx = idx;
            int qhIdx = idx + 128;
            int scalesIdx = idx + 192;

            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int ql0 = src[qlIdx + l] & 0xFF;
                    int ql32 = src[qlIdx + l + 32] & 0xFF;
                    int qhl = src[qhIdx + l] & 0xFF;

                    int q1 = ((ql0 & 0xF) | (((qhl >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql32 & 0xF) | (((qhl >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql0 >> 4) | (((qhl >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql32 >> 4) | (((qhl >> 6) & 3) << 4)) - 32;

                    dst[outIdx + l + 0]  = d * src[scalesIdx + is + 0] * q1;
                    dst[outIdx + l + 32] = d * src[scalesIdx + is + 2] * q2;
                    dst[outIdx + l + 64] = d * src[scalesIdx + is + 4] * q3;
                    dst[outIdx + l + 96] = d * src[scalesIdx + is + 6] * q4;
                }
                outIdx += 128;
                qlIdx += 64;
                qhIdx += 32;
                scalesIdx += 8;
            }

            idx += 210;
        }
    }

    /**
     * Fast float16 (half precision IEEE 754) to float32 conversion.
     */
    public static float halfToFloat(short h) {
        int bits = h & 0xFFFF;
        int s = (bits >>> 15) & 0x00000001;
        int e = (bits >>> 10) & 0x0000001F;
        int f = (bits) & 0x000003FF;

        if (e == 0) {
            if (f == 0) {
                return Float.intBitsToFloat(s << 31);
            } else {
                // Denormalized
                while ((f & 0x00000400) == 0) {
                    f <<= 1;
                    e -= 1;
                }
                e++;
                f &= ~0x00000400;
            }
        } else if (e == 31) {
            if (f == 0) {
                return Float.intBitsToFloat((s << 31) | 0x7F800000);
            } else {
                return Float.intBitsToFloat((s << 31) | 0x7F800000 | (f << 13));
            }
        }

        e = e + (127 - 15);
        f = f << 13;
        return Float.intBitsToFloat((s << 31) | (e << 23) | f);
    }
}
