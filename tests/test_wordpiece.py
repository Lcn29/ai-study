import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


class WordPieceTest(unittest.TestCase):
    def test_course_training(self):
        # 导入模块只定义函数，不应加载分词器或执行训练。
        with patch("transformers.BertTokenizer.from_pretrained") as load_tokenizer:
            module = runpy.run_path(
                str(Path(__file__).resolve().parents[1] / "src/002/wordpiece.py")
            )
        load_tokenizer.assert_not_called()

        # 使用课程的五词示例，验证 ## 前缀拆分与按得分（而非频率）选对的差异。
        word_freqs = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
        alphabet = module["build_alphabet"](word_freqs)
        self.assertEqual(alphabet, ["##g", "##n", "##s", "##u", "b", "h", "p"])
        vocab, splits = module["initialize_training"](word_freqs, alphabet)
        self.assertEqual(splits["hugs"], ["h", "##u", "##g", "##s"])

        # BPE 会先合并频率最高的 ("##u", "##g")（20 次）；
        # WordPiece 因 "##u" 到处出现而得分低，先合并唯一的无 ##u 对 ("##g", "##s")。
        scores = module["compute_pair_scores"](splits, word_freqs)
        self.assertEqual(module["find_best_pair"](scores), (("##g", "##s"), 1 / 20))

        splits = module["merge_pair"]("##g", "##s", splits, word_freqs)
        self.assertEqual(splits["hugs"], ["h", "##u", "##gs"])

        # 最长匹配：词汇表里有 hug 时 hugs -> ["hug", "##s"]；匹配不到时整词 [UNK]。
        vocab.append("hug")
        self.assertEqual(module["encode_word"]("hugs", vocab), ["hug", "##s"])
        self.assertEqual(module["encode_word"]("mug", vocab), ["[UNK]"])


if __name__ == "__main__":
    unittest.main()
