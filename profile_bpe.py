import cProfile
import pstats
from tests.adapters import run_train_bpe
from pathlib import Path

def run_bpe_training():
    """运行BPE训练的函数"""
    input_path = Path("tests/fixtures/corpus.en")
    
    print("开始BPE训练...")
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
    print(f"训练完成！词汇表大小: {len(vocab)}, 合并规则数量: {len(merges)}")
    return vocab, merges

if __name__ == "__main__":
    # 使用 cProfile 进行性能分析
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 运行BPE训练
    vocab, merges = run_bpe_training()
    
    profiler.disable()
    
    # 打印性能统计信息
    print("\n" + "="*50)
    print("性能分析结果:")
    print("="*50)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')  # 按累计时间排序
    stats.print_stats(20)  # 打印前20个最耗时的函数
    
    print("\n" + "="*50)
    print("按调用次数排序:")
    print("="*50)
    stats.sort_stats('calls')
    stats.print_stats(20)  # 打印前20个调用次数最多的函数
