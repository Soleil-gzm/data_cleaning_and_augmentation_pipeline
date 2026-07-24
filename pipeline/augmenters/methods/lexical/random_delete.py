"""
随机删除词增强器（lexical）

支持保护词列表，避免删除对模型训练重要的语气词和确认词。

配置示例：
    augmenters:
      random_delete:
        weight: 1.0
        prob: 0.1              # 默认删除概率
        protected_words:       # 保护词列表（不参与随机删除）
          - "嗯"
          - "嗯嗯"
          - "是"
          - "是的"
          - "对"
          - "对的"
          - "没错"
          - "好的"
          - "行"
          - "收到"
          - "了解"
"""
from typing import Optional

from ...base import BaseAugmenter
from ...utils import tokenize
from ....utils.random_utils import rand


class RandomDeleteAugmenter(BaseAugmenter):
    DEFAULT_PROTECTED_WORDS = {
        "嗯", "没错", "对。",
        "是", "我是", "是的", "对的", "肯定",
        "好", "好的", "好嘞",
        "行", "可以", "没问题",
        "收到", "了解", "明白",
        "啊", "呀", "哦", "噢", "喔", "呢", "吧", "嘛", "咯",
        "肯", "定", "是我",
    }

    def __init__(self, config):
        super().__init__(config)
        self.prob = float(config.get("prob", 0.1))
        
        # 保护词列表：配置中的 + 默认的
        protected_config = config.get("protected_words", [])
        self.protected_words = set(self.DEFAULT_PROTECTED_WORDS)
        if protected_config:
            self.protected_words.update(protected_config)
        
        # 将保护词添加到 jieba 自定义词典，防止被拆分
        self._add_protected_words_to_jieba()

    def _add_protected_words_to_jieba(self):
        """将保护词添加到 jieba 自定义词典，防止被拆分"""
        try:
            import jieba
            for word in self.protected_words:
                if len(word) >= 2:
                    jieba.add_word(word, freq=10000)
        except ImportError:
            pass

    def apply(self, text: str, rng=None) -> str:
        if not isinstance(text, str) or not text.strip():
            return text
        tokens = tokenize(text)
        if len(tokens) < 3:
            return text
        
        new_tokens = []
        for token in tokens:
            # 保护词直接保留，不参与随机删除
            if token in self.protected_words:
                new_tokens.append(token)
                continue
            # 非保护词按概率删除
            if rand(rng=rng) > self.prob:
                new_tokens.append(token)
        
        if not new_tokens or len(new_tokens) < 2:
            return text
        return ''.join(new_tokens)
