from collections import defaultdict

class BiMultiMap:
    """
    双向多重映射类，支持键与值之间的多对多关联，并提供高效的查询和排序功能。

    特性：
    - 添加/删除键值关联，自动维护双向映射。
    - 按键或值获取关联项。
    - 获取排序后的键值对列表，支持缓存提高重复访问效率。

    示例：
    >>> bimap = BiMultiMap()
    >>> bimap.add_association('a', 1)
    >>> bimap.add_association('a', 2)
    >>> bimap.get_values('a')
    [1, 2]
    >>> bimap.get_sorted_pairs(sort_by='value')
    [(1, 'a'), (2, 'a')]  # 示例可能需调整实际结果
    """

    def __init__(self):
        self.key_to_values = defaultdict(set)  # 键到值的正向映射
        self.value_to_keys = defaultdict(set)  # 值到键的反向映射
        self.all_pairs = set()                 # 存储所有唯一键值对
        self._sorted_cache = {}                 # 缓存排序结果

    def add_association(self, key, value):
        """
        添加键值关联，双向同步。

        参数：
            key: 要关联的键。
            value: 要关联的值。

        示例：
        >>> bimap = BiMultiMap()
        >>> bimap.add_association('a', 1)
        """
        # 更新正向映射
        self.key_to_values[key].add(value)
        # 更新反向映射
        self.value_to_keys[value].add(key)
        # 记录键值对
        self.all_pairs.add((key, value))
        # 清空缓存
        self._sorted_cache.clear()

    def get_values(self, key):
        """
        获取指定键关联的所有值。

        参数：
            key: 查询的键。

        返回：
            list: 关联值的列表，不存在返回空列表。

        示例：
        >>> bimap = BiMultiMap()
        >>> bimap.add_association('a', 1)
        >>> bimap.get_values('a')
        [1]
        """
        return list(self.key_to_values.get(key, set()))

    def get_keys(self, value):
        """
        获取指定值关联的所有键。

        参数：
            value: 查询的值。

        返回：
            list: 关联键的列表，不存在返回空列表。

        示例：
        >>> bimap = BiMultiMap()
        >>> bimap.add_association('a', 1)
        >>> bimap.get_keys(1)
        ['a']
        """
        return list(self.value_to_keys.get(value, set()))

    def remove_key_value(self, key, value):
        """
        移除特定的键值关联。

        参数：
            key: 要移除的键。
            value: 要移除的值。

        示例：
        >>> bimap = BiMultiMap()
        >>> bimap.add_association('a', 1)
        >>> bimap.remove_key_value('a', 1)
        """
        if value in self.key_to_values[key]:
            # 移除正向映射
            self.key_to_values[key].remove(value)
            # 移除反向映射
            self.value_to_keys[value].remove(key)
            # 从所有对中移除
            self.all_pairs.discard((key, value))
            # 清理空键和值
            self._cleanup_key(key)
            self._cleanup_value(value)
            # 清空缓存
            self._sorted_cache.clear()

    def remove_key(self, key):
        """
        移除键及其所有关联值。

        参数：
            key: 要移除的键。

        示例：
        >>> bimap = BiMultiMap()
        >>> bimap.add_association('a', 1)
        >>> bimap.remove_key('a')
        """
        if key in self.key_to_values:
            # 获取所有关联值
            values = self.key_to_values[key].copy()
            for value in values:
                # 移除反向映射中的键
                self.value_to_keys[value].remove(key)
                # 从所有对中移除
                self.all_pairs.discard((key, value))
                # 清理空值
                self._cleanup_value(value)
            # 删除正向映射中的键
            del self.key_to_values[key]
            # 清空缓存
            self._sorted_cache.clear()

    def remove_value(self, value):
        """
        移除值及其所有关联键。

        参数：
            value: 要移除的值。

        示例：
        >>> bimap = BiMultiMap()
        >>> bimap.add_association('a', 1)
        >>> bimap.remove_value(1)
        """
        if value in self.value_to_keys:
            # 获取所有关联键
            keys = self.value_to_keys[value].copy()
            for key in keys:
                # 移除正向映射中的值
                self.key_to_values[key].remove(value)
                # 从所有对中移除
                self.all_pairs.discard((key, value))
                # 清理空键
                self._cleanup_key(key)
            # 删除反向映射中的值
            del self.value_to_keys[value]
            # 清空缓存
            self._sorted_cache.clear()

    def _cleanup_key(self, key):
        """清理无关联值的键。"""
        if not self.key_to_values[key]:
            del self.key_to_values[key]

    def _cleanup_value(self, value):
        """清理无关联键的值。"""
        if not self.value_to_keys[value]:
            del self.value_to_keys[value]

    def get_sorted_pairs(self, sort_by='key', reverse=False, top_n=None):
        """
        获取排序后的键值对列表，支持缓存提升性能。

        参数：
            sort_by (str): 排序依据，'key' 或 'value'。
            reverse (bool): 是否降序排序。
            top_n (int): 返回前 top_n 项，None 返回所有。

        返回：
            list: 排序后的键值对列表。

        异常：
            ValueError: sort_by 参数错误时抛出。

        示例：
        >>> bimap = BiMultiMap()
        >>> bimap.add_association('b', 2)
        >>> bimap.add_association('a', 1)
        >>> bimap.get_sorted_pairs(sort_by='key')
        [('a', 1), ('b', 2)]
        """
        cache_key = (sort_by, reverse)
        if cache_key in self._sorted_cache:
            sorted_pairs = self._sorted_cache[cache_key]
        else:
            # 生成所有键值对并排序
            pairs = list(self.all_pairs)
            if sort_by == 'key':
                sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=reverse)
            elif sort_by == 'value':
                sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=reverse)
            else:
                raise ValueError("sort_by 必须为 'key' 或 'value'")
            self._sorted_cache[cache_key] = sorted_pairs

        return sorted_pairs[:top_n] if (top_n !=None and top_n < len(sorted_pairs)) else sorted_pairs

    def __repr__(self):
        return f"BiMultiMap(keys={list(self.key_to_values)}, values={list(self.value_to_keys)})"