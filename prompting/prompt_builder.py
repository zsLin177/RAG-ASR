from typing import List


def build_prompt(query: str, keywords: List[str]) -> str:
    keywords_str = '\n'.join(f"- {k}" for k in keywords)
    prompt = (
        "请根据以下信息，输出更好的文本结果。\n"
        "任务：纠正并完善ASR候选文本，使其符合语义、拼写与领域关键词。\n\n"
        f"ASR候选：{query}\n\n"
        "可能相关的关键词（供参考）：\n"
        f"{keywords_str}\n\n"
        "请输出修正后的最终文本，仅输出文本本身。"
    )
    return prompt 