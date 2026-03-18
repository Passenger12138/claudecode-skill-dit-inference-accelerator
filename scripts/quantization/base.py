"""
基础设施：SDOps 和 ModuleOps

用于在加载模型时进行状态字典转换和模块变换。
"""

from dataclasses import dataclass, replace
from typing import Callable, NamedTuple, Protocol

import torch


class KeyValueOperationResult(NamedTuple):
    """状态字典 key-value 操作结果"""

    new_key: str
    new_value: torch.Tensor


@dataclass(frozen=True, slots=True)
class ContentReplacement:
    """用于替换状态字典 key 中的内容"""

    content: str
    replacement: str


@dataclass(frozen=True, slots=True)
class ContentMatching:
    """用于匹配状态字典 key 的前缀和后缀"""

    prefix: str = ""
    suffix: str = ""


class KeyValueOperation(Protocol):
    """key-value 操作协议"""

    def __call__(
        self, tensor_key: str, tensor_value: torch.Tensor
    ) -> list[KeyValueOperationResult]: ...


@dataclass(frozen=True, slots=True)
class SDKeyValueOperation:
    """状态字典 key-value 操作"""

    key_matcher: ContentMatching
    kv_operation: KeyValueOperation


@dataclass(frozen=True, slots=True)
class SDOps:
    """
    不可变的状态字典操作类

    用于在加载模型权重时进行转换（如量化）。
    """

    name: str
    mapping: tuple[ContentReplacement | ContentMatching | SDKeyValueOperation, ...] = ()

    def with_replacement(self, content: str, replacement: str) -> "SDOps":
        """创建新实例，添加字符串替换操作"""
        new_mapping = (*self.mapping, ContentReplacement(content, replacement))
        return replace(self, mapping=new_mapping)

    def with_matching(self, prefix: str = "", suffix: str = "") -> "SDOps":
        """创建新实例，添加前缀/后缀匹配"""
        new_mapping = (*self.mapping, ContentMatching(prefix, suffix))
        return replace(self, mapping=new_mapping)

    def with_kv_operation(
        self,
        operation: KeyValueOperation,
        key_prefix: str = "",
        key_suffix: str = "",
    ) -> "SDOps":
        """创建新实例，添加 key-value 操作"""
        key_matcher = ContentMatching(key_prefix, key_suffix)
        sd_kv_operation = SDKeyValueOperation(key_matcher, operation)
        new_mapping = (*self.mapping, sd_kv_operation)
        return replace(self, mapping=new_mapping)

    def apply_to_key(self, key: str) -> str | None:
        """应用映射到 key"""
        matchers = [c for c in self.mapping if isinstance(c, ContentMatching)]
        valid = any(
            key.startswith(f.prefix) and key.endswith(f.suffix) for f in matchers
        )
        if not valid:
            return None

        for replacement in self.mapping:
            if not isinstance(replacement, ContentReplacement):
                continue
            if replacement.content in key:
                key = key.replace(replacement.content, replacement.replacement)
        return key

    def apply_to_key_value(
        self, key: str, value: torch.Tensor
    ) -> list[KeyValueOperationResult]:
        """应用操作到指定的 key-value"""
        for operation in self.mapping:
            if not isinstance(operation, SDKeyValueOperation):
                continue
            if key.startswith(operation.key_matcher.prefix) and key.endswith(
                operation.key_matcher.suffix
            ):
                return operation.kv_operation(key, value)
        return [KeyValueOperationResult(key, value)]


class ModuleOps(NamedTuple):
    """
    模块操作：用于匹配和变换 PyTorch 模块

    - matcher: 判断模块是否需要变换
    - mutator: 执行变换操作
    """

    name: str
    matcher: Callable[[torch.nn.Module], bool]
    mutator: Callable[[torch.nn.Module], torch.nn.Module]
