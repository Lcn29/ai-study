from transformers import GPT2Tokenizer, TokenizersBackend
from collections import defaultdict

# 源码来源：https://huggingface.co/learn/llm-course/zh-CN/chapter6/5
# BPE 的核心思路：从最小符号开始，反复把最常见的相邻词元合成一个新词元。
# 词元不一定是完整单词，也可以是单个字符、单词的一部分或带空格标记的片段。
# 这里的“训练”是学习分词规则，不是训练语言模型的神经网络参数。
# 本示例只借用 GPT-2 的预分词器，不使用它已有的 BPE 合并规则来完成后续分词。
#
# 阅读时先区分以下数据，避免把“训练片段”误认为“最终词元”：
# word_freqs：片段 -> 出现次数；保存训练数据的频率，不是最终词汇表。
# alphabet：训练片段中出现的基础字符；作为合并的起点。
# splits：片段 -> 当前词元列表；随着合并不断变化，片段本身的出现次数不变。
# merges：相邻词元对 -> 合并后的词元；按学习顺序记录，供新文本分词时重放。
# vocab：可用词元的列表；包含特殊词元、基础字符和每次合并生成的词元。
#
# 这是教学实现：只收集语料中出现的基础字符，没有覆盖完整字节表，
# 也没有实现未知字符处理、词元到整数编号的转换或大型语料的高效训练。

# 先划分片段再统计频率，让相同片段只保存一份，后续通过频率还原其统计权重。
# 这样不必为每次出现都维护一份相同的拆分结果，但也不是把这些片段直接加入词汇表。
def compute_word_freqs(corpus, tokenizer: TokenizersBackend):
    # 首次遇到某个片段时默认计数为 0，因此可以直接累加，不必先判断键是否存在。
    word_freqs = defaultdict(int)

    for text in corpus:
        # 预分词只是先划定处理边界，还没有应用本示例要学习的 BPE 规则。
        # 后续只在一个片段内部合并，因此不能把两个不同片段的末尾和开头拼在一起。
        # 这里的 word 是预分词片段，不一定是自然语言中的完整单词，标点也能成为片段。
        # GPT-2 将空格字节映射为可见字符 Ġ，例如 Ġis 对应原文的 " is"，解码时会还原为空格。
        # 不丢弃这个标记，才能保留“前面有没有空格”的区别。
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        # 每项包含片段和原文中的起止位置；本例只需要频率，不需要定位，所以舍弃位置。
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1
    return word_freqs


# 从最小单位起步，而不是一开始就把完整单词都放进词汇表。
# 这样训练才能逐步发现可复用的子词，而不是只能识别语料中见过的完整单词。
# 这里收集的是预分词输出中的字符，包括 Ġ，而不是直接遍历原文的可见字母。
def build_alphabet(word_freqs):
    alphabet = []

    for word in word_freqs.keys():
        # 只需遍历不重复的片段：某个片段出现多少次，不影响它包含哪些基础字符。
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    # 排序只为了让基础词汇表的展示顺序稳定，不是在决定后续合并的优先级。
    alphabet.sort()
    return alphabet


# vocab 回答“目前有哪些词元”，splits 回答“每个训练片段目前由哪些词元组成”。
# 初始拆分只含单个字符；随着训练推进，列表中的元素会逐步变成更长的子词。
def initialize_training(word_freqs, alphabet):
    # 预留 GPT-2 的文本结束词元；它没有被放入 splits，因此不参与普通片段的合并。
    # 创建独立列表，使后续扩充 vocab 时不改变基础字符表 alphabet。
    vocab = ["<|endoftext|>"] + alphabet.copy()
    # 例如测试中的 hug 初始拆成 ['h', 'u', 'g']，而不是一开始就当作一个词元。
    splits = {word: [c for c in word] for word in word_freqs.keys()}
    return vocab, splits


# BPE 需要比较的是“相邻词元对”的频率，而不是完整片段的频率。
# 只统计相邻对，因为合并必须保留原文顺序，不能越过中间的词元拼接。
def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        # 只剩一个词元时，不存在可以合并的相邻对。
        if len(split) == 1:
            continue
        # 每次取当前位置和下一个位置，因此最后一个位置不能作为一对的起点。
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            # splits 只保存片段的一份拆分，所以这里必须加 freq，而不是只加 1。
            # 例如 hug 出现 10 次，初始拆分中的 ('h', 'u') 就贡献 10 次计数。
            # 同一相邻对在一个片段中出现多处时，每个位置都要分别累计 freq。
            pair_freqs[pair] += freq
    return pair_freqs


# 每轮只选当前最常见的相邻对，让新词元优先覆盖语料中高频出现的组合。
# 这是逐轮做局部选择，不是在提前搜索一套全局最优的合并规则。
def find_best_pair(pair_freqs):
    best_pair = ""
    max_freq = None

    for pair, freq in pair_freqs.items():
        # None 表示还没选过任何一对；选过之后，只有更高频的对才能替换当前结果。
        # 使用严格小于号，所以频率相同时保留最先遍历到的一对。
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    # 频率表为空时会返回 ("", None)；train_bpe 没有处理规则耗尽的情况，见其说明。
    return best_pair, max_freq


# 在所有训练片段中，将相邻的 a 和 b 替换成一个新词元 a + b。
# 随着训练推进，a 和 b 也可以是之前合并出的词元，从而形成更长的子词。
# 同一条规则要更新所有片段，不能只更新贡献了最高频计数的某一个片段。
def merge_pair(a, b, splits, word_freqs):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # 保留左边 + 放入新词元 + 保留右边：两个相邻元素被替换成一个元素。
                # 字符内容和先后顺序没变，变化的只是词元边界，列表长度因此减少 1。
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                # 切片合并后列表变短，索引位置也随之改变，所以用 while 控制扫描。
                # 未合并时才向后移动；合并后重新检查当前位置，再继续向右扫描。
                i += 1
        # 切片拼接得到的是新列表，必须写回字典，下一轮统计才能看到新的词元边界。
        splits[word] = split
    return splits


# 不断学习最高频的合并规则，直到词汇表达到目标大小。
# 词汇表大小限制本例训练多少轮，而不是要求每个片段最终都合成完整单词。
# ponytail: 本例未处理相邻对耗尽；用于更小语料或更大目标词表前，需补充空频率表退出条件。
def train_bpe(splits, word_freqs, vocab, merges, vocab_size):
    while len(vocab) < vocab_size:
        # 合并后旧的相邻对消失，也会出现新的相邻对，不能一直使用最初的频率表。
        # 例如 hug 从 ['h', 'u', 'g'] 变成 ['h', 'ug'] 后，要开始统计 ('h', 'ug')。
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        best_pair, max_freq = find_best_pair(pair_freqs)
        # *best_pair 把二元组展开成两个实参，分别传给 merge_pair 的 a 和 b。
        splits = merge_pair(*best_pair, splits, word_freqs)
        # splits 更新训练状态；merges 保存以后如何分词；vocab 登记新增的可用词元。
        # 三者各有用途：只有最终词汇表，不能直接说明这些词元是按什么顺序合并出来的。
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])

    # 上面还直接更新了传入的 splits 字典；即使不返回它，调用方也能看到新的拆分。
    return vocab, merges


# 对新文本使用相同的预分词方式，再按训练顺序应用已学到的合并规则。
# 此时不再统计频率或学习规则，否则同一文本的分词方式会随输入内容改变。
# 本函数返回词元字符串，而不是语言模型实际接收的整数编号。
# 遇到训练字符表中没有的字符时，本例会保留该字符，却不检查它是否属于 vocab。
def tokenize(text, tokenizer: TokenizersBackend, merges):
    # 训练与使用时必须采用相同的预分词方式，才能保持片段边界和空格表示一致。
    pre_tokenize_result = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    # 新文本可能包含训练时没有出现过的片段，所以不能直接查询训练用的 splits。
    # 从字符重新开始，再重放规则，就能使用已有子词处理新片段。
    splits = [[l for l in word] for word in pre_tokenized_text]
    # merges 字典保留插入顺序：先学到的规则必须先执行，后续规则才有输入可合并。
    # 例如 ('u', 'g') -> 'ug' 必须先于 ('h', 'ug') -> 'hug' 执行。
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            # 合并只发生在当前片段内；写回后，下一条规则继续基于更新后的列表处理。
            splits[idx] = split

    # 此处的 sum 不是数值求和：从空列表开始，依次拼接各片段的词元列表。
    # 只展平列表，不把词元重新连成字符串，因此保留最终的词元边界和文本顺序。
    return sum(splits, [])


# 按“准备数据 -> 初始化 -> 学习规则 -> 使用规则”的顺序组织演示。
# 先单独展示首次合并，再循环训练，便于观察一次合并具体改变了哪些数据。
def main():
    # 阶段 1：先构造一个包含四句话的小型语料库，用于演示训练过程。
    print("[阶段 1] 准备训练语料")
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    print("训练语料：", corpus)

    # 阶段 2：使用 GPT-2 的预分词器划分语料，同时统计每个片段的出现次数。
    # 此处只借用预分词步骤，后面的 BPE 合并规则由本示例自行学习。
    # 加载 GPT-2 分词器不等于加载 GPT-2 语言模型；这里需要的是它的预分词组件。
    print("\n[阶段 2] 预分词并统计词频")
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # 后续要访问 backend_tokenizer，先检查后端类型，避免到处理语料时才访问失败。
    if not isinstance(tokenizer, TokenizersBackend):
        raise TypeError("本示例需要 TokenizersBackend 类型的分词器。")
    word_freqs = compute_word_freqs(corpus, tokenizer)
    print("词频统计：", word_freqs)

    # 阶段 3：从所有训练片段中提取不重复的字符，作为基础词汇表的组成部分。
    # 片段出现得多不代表基础字符要重复登记；重复次数只用于后面的相邻对统计。
    print("\n[阶段 3] 构建基础字符表")
    alphabet = build_alphabet(word_freqs)
    print(f"字符表（{len(alphabet)} 个字符）：", alphabet)

    # 阶段 4：加入 GPT-2 的特殊词元，再将每个片段拆成单个字符以开始训练。
    # Ġtrained 的输出用于跟踪同一个片段，观察合并前后词元边界的变化。
    print("\n[阶段 4] 初始化词汇表和字符拆分")
    vocab, splits = initialize_training(word_freqs, alphabet)
    print(f"初始词汇表（{len(vocab)} 个词元）：", vocab)
    print("Ġtrained 的初始拆分：", splits["Ġtrained"])

    # 阶段 5：统计同一片段内相邻词元对的频率，并按片段在语料中的词频加权。
    # 只展示遍历顺序中的前六组，并非频率最高的六组；最高频对由下一阶段单独选择。
    print("\n[阶段 5] 计算相邻词元对的频率")
    pair_freqs = compute_pair_freqs(splits, word_freqs)
    for i, key in enumerate(pair_freqs.keys()):
        print(f"{key}: {pair_freqs[key]}")
        if i >= 5:
            break

    # 阶段 6：选择最常见的相邻对，记录规则、扩充词汇表，并更新所有训练片段。
    # 本语料的首次合并为 ("Ġ", "t") -> "Ġt"，即把空格与字母 t 合成一个词元。
    # 下面为了展示过程手动写出这条规则，并没有用 best_pair 动态完成首次合并；
    # 如果更换语料，必须同步调整这部分，不能认为最高频对总是 ("Ġ", "t")。
    print("\n[阶段 6] 学习并应用首次合并规则")
    best_pair, max_freq = find_best_pair(pair_freqs)
    print(f"最高频相邻对：{best_pair}，出现 {max_freq} 次")

    merges = {("Ġ", "t"): "Ġt"}
    vocab.append("Ġt")
    splits = merge_pair("Ġ", "t", splits, word_freqs)
    print("首次合并规则：", merges)
    print("Ġtrained 合并后的拆分：", splits["Ġtrained"])

    # 阶段 7：重复统计频率、选择最高频对和合并，直到词汇表大小达到 50。
    # 最终包含初始的 31 个词元和 19 条合并规则生成的新词元。
    # 首次合并已经执行过，这里接着训练，而不是重新初始化；50 只是本演示的目标。
    print("\n[阶段 7] 继续训练至目标词汇表大小")
    vocab, merges = train_bpe(splits, word_freqs, vocab, merges, 50)
    print(f"训练完成：词汇表包含 {len(vocab)} 个词元，共学习 {len(merges)} 条合并规则")
    print("合并规则：", merges)
    print("最终词汇表：", vocab)

    # 阶段 8：对新文本预分词并拆成字符，再按学习顺序应用全部合并规则。
    # 结果：['This', 'Ġis', 'Ġ', 'n', 'o', 't', 'Ġa', 'Ġtoken', '.']。
    # 可以看到词元长度并不一致：高频组合能合成较长词元，其他部分仍可能按字符保留。
    print("\n[阶段 8] 使用合并规则对新文本分词")
    print("待分词文本：This is not a token.")
    print("分词结果：", tokenize("This is not a token.", tokenizer, merges))


# 直接运行文件才启动演示；被测试或其他模块加载时，只定义函数，不加载分词器或训练。
if __name__ == "__main__":
    main()
