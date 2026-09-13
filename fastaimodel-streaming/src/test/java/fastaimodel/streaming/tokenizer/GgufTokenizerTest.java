package fastaimodel.streaming.tokenizer;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class GgufTokenizerTest {

    @Test
    public void testSentencePieceEncodingAndDecoding() {
        List<String> vocab = Arrays.asList("<unk>", "<s>", "</s>", "▁Hallo", "▁Welt", "!", "▁Quanten", "physik");
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
}
