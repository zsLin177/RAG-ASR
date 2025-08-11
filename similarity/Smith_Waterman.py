import re
from Bio import Align

class SimpleCharacterAligner:
    """
    一个极简的字符级别对齐器。
    【最稳定版】：对所有输入都作为字符串，在字符级别进行Smith-Waterman比对。
    不区分中英文逻辑，保证100%运行。
    """

    def __init__(self):
        """
        初始化一个通用的、使用固定分值的打分器。
        """
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = 'local'  # Smith-Waterman
        
        # 设定一套通用的、固定的评分规则
        self.match_score = 5
        self.mismatch_score = -4
        self.open_gap_score = -8
        self.extend_gap_score = -1
        
        self.aligner.match_score = self.match_score
        self.aligner.mismatch_score = self.mismatch_score
        self.aligner.open_gap_score = self.open_gap_score
        self.aligner.extend_gap_score = self.extend_gap_score
        
        print("极简字符打分器已初始化 (100% 稳定版)。")

    def align(self, text1, text2, normalize=True):
        """
        对两个字符串进行字符级别的比对。

        :param text1: 第一个字符串。
        :param text2: 第二个字符串。
        :param normalize: 是否将结果归一化到 0-1 之间。
        :return: 包含得分信息的字典。
        """
        # 为保证一致性，全部转为小写
        seq1 = text1.lower()
        seq2 = text2.lower()
        
        if not seq1 or not seq2:
            return {'raw_score': 0, 'normalized_score': 0, 'alignment': None}
            
        # 直接对字符串进行比对
        alignments = self.aligner.align(seq2, seq1)

        if not alignments:
            return {'raw_score': 0, 'normalized_score': 0, 'alignment': None}

        best_alignment = alignments[0]
        raw_score = best_alignment.score

        # 归一化处理
        normalized_score = 0
        if normalize:
            shorter_len = min(len(seq1), len(seq2))
            if shorter_len > 0:
                max_possible_score = shorter_len * self.match_score
                normalized_score = max(0, raw_score) / max_possible_score
        
        return {
            'raw_score': raw_score,
            'normalized_score': round(normalized_score, 4),
            'alignment': best_alignment
        }

# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 初始化类
    aligner = SimpleCharacterAligner()
    print("-" * 50)

    # 2. 英文场景示例
    #    请注意：现在比较的是单个字母，包括空格！
    print(">>> 英文场景测试 (字符级别)")
    asr_en = "show me the wether"
    keyword_en = "show me weather"
    result_en = aligner.align(asr_en, keyword_en)
    
    print(f"文本1: '{asr_en}'")
    print(f"文本2: '{keyword_en}'")
    print(f"原始得分: {result_en['raw_score']}")
    print(f"归一化相似度: {result_en['normalized_score']}")
    if result_en['alignment']:
        print("比对详情:\n" + str(result_en['alignment']))
    
    print("-" * 50)

    # 3. 中文场景示例 (字符级别)
    print(">>> 中文场景测试 (字符级别)")
    asr_zh = "我想去付兴路呀"
    keyword_zh = "天下第一呀"
    result_zh = aligner.align(asr_zh, keyword_zh)
    
    print(f"文本1: '{asr_zh}'")
    print(f"文本2: '{keyword_zh}'")
    print(f"原始得分: {result_zh['raw_score']}")
    print(f"归一化相似度: {result_zh['normalized_score']}")
    if result_zh['alignment']:
        print("比对详情:\n" + str(result_zh['alignment']))