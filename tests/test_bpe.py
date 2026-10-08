import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


class BpeTest(unittest.TestCase):
    def test_course_training(self):
        # 导入模块只定义函数，不应加载分词器或执行训练。
        with patch("transformers.GPT2Tokenizer.from_pretrained") as load_tokenizer:
            module = runpy.run_path(
                str(Path(__file__).resolve().parents[1] / "src/001/bpe.py")
            )
        load_tokenizer.assert_not_called()

        # 使用课程的五词示例，验证加权频率和前三条合并规则的学习顺序。
        word_freqs = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
        alphabet = module["build_alphabet"](word_freqs)
        self.assertEqual(alphabet, ["b", "g", "h", "n", "p", "s", "u"])
        vocab, splits = module["initialize_training"](word_freqs, alphabet)
        pair_freqs = module["compute_pair_freqs"](splits, word_freqs)
        self.assertEqual(module["find_best_pair"](pair_freqs), (("u", "g"), 20))

        vocab, merges = module["train_bpe"](splits, word_freqs, vocab, {}, 11)
        self.assertEqual(len(vocab), 11)
        self.assertEqual(
            list(merges.items()),
            [(("u", "g"), "ug"), (("u", "n"), "un"), (("h", "ug"), "hug")],
        )
        self.assertEqual(splits["hugs"], ["hug", "s"])

    def test_missing_tokenizer(self):
        module = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "src/001/bpe.py")
        )
        # 加载结果为空时，应在预分词前给出明确的后端类型错误。
        with (
            patch("transformers.GPT2Tokenizer.from_pretrained", return_value=None),
            patch("builtins.print"),
            self.assertRaisesRegex(TypeError, "TokenizersBackend"),
        ):
            module["main"]()


if __name__ == "__main__":
    unittest.main()
