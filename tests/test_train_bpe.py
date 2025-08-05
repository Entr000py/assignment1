import json # 导入json模块，用于处理JSON数据
import time # 导入time模块，用于时间测量

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    确保BPE训练相对高效，通过测量在小数据集上的训练时间，
    如果超过1.5秒则抛出错误。
    这是一个相当宽松的上限，在我的笔记本上，参考实现需要0.38秒。
    相比之下，玩具实现大约需要3秒。
    """
    input_path = FIXTURES_PATH / "corpus.en" # 定义输入文件路径
    start_time = time.time()
    _, _ = run_train_bpe( # 运行BPE训练
        input_path=input_path, # 输入文件路径
        vocab_size=500, # 词汇表大小
        special_tokens=["<|endoftext|>"], # 特殊token
    )
    end_time = time.time()
    assert end_time - start_time < 1.5 # 断言训练时间小于1.5秒


def test_train_bpe():
    # 定义输入文件路径
    input_path = FIXTURES_PATH / "corpus.en"
    # 运行BPE训练，获取词汇表和合并规则
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # 参考分词器词汇表和合并规则的路径
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # 比较学习到的合并规则与预期的输出合并规则
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()} # 创建GPT2字节解码器
    with open(reference_merges_path, encoding="utf-8") as f: # 打开参考合并文件
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f] # 读取并解析GPT2参考合并规则
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]), # 解码第一个合并token
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]), # 解码第二个合并token
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges # 遍历GPT2参考合并规则
        ]
    assert merges == reference_merges # 断言合并规则匹配

    # 比较词汇表与预期的输出词汇表
    with open(reference_vocab_path, encoding="utf-8") as f: # 打开参考词汇表文件
        gpt2_reference_vocab = json.load(f) # 加载GPT2参考词汇表
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item]) # 解码词汇项
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items() # 遍历GPT2参考词汇表
        }
    # 不检查词汇表是否完全匹配（因为它们可能以不同方式构建），
    # 而是确保词汇表的键和值匹配
    assert set(vocab.keys()) == set(reference_vocab.keys()) # 断言词汇表的键集合匹配
    assert set(vocab.values()) == set(reference_vocab.values()) # 断言词汇表的值集合匹配


def test_train_bpe_special_tokens(snapshot):
    """
    确保特殊token被添加到词汇表中，并且不与其他token合并。
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt" # 定义输入文件路径
    vocab, merges = run_train_bpe( # 运行BPE训练
        input_path=input_path, # 输入文件路径
        vocab_size=1000, # 词汇表大小
        special_tokens=["<|endoftext|>"], # 特殊token
    )

    # 检查特殊token是否不在词汇表中
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"] # 过滤掉特殊token
    for word_bytes in vocabs_without_specials: # 遍历非特殊token
        assert b"<|" not in word_bytes # 断言不包含特殊token的起始标记

    snapshot.assert_match( # 使用snapshot断言匹配
        {
            "vocab_keys": set(vocab.keys()), # 词汇表的键集合
            "vocab_values": set(vocab.values()), # 词汇表的值集合
            "merges": merges, # 合并规则
        },
    )
