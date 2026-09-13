package fastaimodel.streaming.tokenizer;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class GgufTokenizerTest {

    @Test
    public void testSentencePieceEncodingAndDecoding() {
        List<String> vocab = Arrays.asList("<unk>", "<s>", "</s>", "\u2581Hallo", "\u2581Welt", "!", "\u2581Quanten", "physik");
        float[] scores = new float[vocab.size()];
        GgufTokenizer tokenizer = new GgufTokenizer(vocab, scores, 1, 2, 0, true, "llama");

        List<Integer> encoded = tokenizer.encode("Hallo Welt!");
        assertNotNull(encoded);
        assertEquals(1, encoded.get(0)); // BOS

        StringBuilder sb = new StringBuilder();
        for (int i = 1; i < encoded.size(); i++) {
            sb.append(tokenizer.decode(encoded.get(i)));
        }
        assertEquals(" Hallo Welt!", sb.toString());
    }

    @Test
    public void testByteBpeChatTokensAndUmlauts() {
        List<String> vocab = Arrays.asList(
                "<|endoftext|>", "<|im_start|>", "<|im_end|>", "user", "\u010a", "H", "all", "o",
                "\u0120W", "elt", "!", "\u0120\u00c3\u00a4"
        );
        float[] scores = new float[vocab.size()];
        GgufTokenizer tokenizer = new GgufTokenizer(vocab, scores, 1, 2, 0, false, "smollm");

        assertTrue(tokenizer.isStopToken(0));
        assertTrue(tokenizer.isStopToken(2));

        String prompt = "<|im_start|>user\nHallo Welt!<|im_end|>";
        List<Integer> encoded = tokenizer.encode(prompt);
        assertNotNull(encoded);
        assertEquals(1, encoded.get(0)); // <|im_start|>
        assertEquals(3, encoded.get(1)); // user
        assertEquals(4, encoded.get(2)); // \n (\u010a)
        assertEquals(2, encoded.get(encoded.size() - 1)); // <|im_end|>
    }
}
