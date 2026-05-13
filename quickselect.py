"""
快速选择算法 (Quick Select) - 查找数组中第 k 个最大的数

快速选择算法是快速排序的变体，用于在未排序的数组中找到第 k 小（或第 k 大）的元素。
平均时间复杂度：O(n)
最坏时间复杂度：O(n²)
空间复杂度：O(1)（原地分区）
"""

import random
from typing import List, Union


def quickselect_kth_largest(nums: List[Union[int, float]], k: int) -> Union[int, float]:
    """
    使用快速选择算法查找数组中第 k 个最大的数。
    
    参数:
        nums: 输入的整数或浮点数列表
        k: 要查找的第 k 个最大数（1-based，k=1 表示最大数）
    
    返回:
        第 k 个最大的数
    
    异常:
        ValueError: 当 k 超出有效范围时抛出
        TypeError: 当输入类型不正确时抛出
    
    示例:
        >>> quickselect_kth_largest([3, 2, 1, 5, 6, 4], 2)
        5
        >>> quickselect_kth_largest([1, 2, 3, 4, 5], 1)
        5
        >>> quickselect_kth_largest([3, 2, 1, 5, 6, 4], 5)
        2
    """
    # 边界情况处理：空数组
    if not nums:
        raise ValueError("输入数组不能为空")
    
    # 边界情况处理：k 超出范围
    n = len(nums)
    if k < 1 or k > n:
        raise ValueError(f"k 必须在 1 到 {n} 之间，当前 k={k}")
    
    # 边界情况处理：类型检查
    if not isinstance(nums, list):
        raise TypeError("nums 必须是列表类型")
    
    # 创建数组副本，避免修改原数组
    arr = nums.copy()
    
    # 第 k 大的数等价于第 (n-k) 小的数（0-based 索引）
    # 例如：第 1 大 = 第 (n-1) 小，第 n 大 = 第 0 小
    target_index = n - k
    
    return _quickselect(arr, 0, n - 1, target_index)


def _quickselect(arr: List[Union[int, float]], left: int, right: int, 
                 target_index: int) -> Union[int, float]:
    """
    快速选择算法的递归实现。
    
    参数:
        arr: 待处理的数组
        left: 当前区间的左边界
        right: 当前区间的右边界
        target_index: 目标元素的索引（0-based）
    
    返回:
        目标位置的元素值
    """
    # 递归终止条件：区间缩小到单个元素
    while left <= right:
        # 分区操作，返回 pivot 的最终位置
        pivot_index = _partition(arr, left, right)
        
        # 如果 pivot 正好在目标位置，返回该元素
        if pivot_index == target_index:
            return arr[pivot_index]
        # 如果目标在左侧，在左半部分继续查找
        elif target_index < pivot_index:
            right = pivot_index - 1
        # 如果目标在右侧，在右半部分继续查找
        else:
            left = pivot_index + 1
    
    # 理论上不会到达这里
    raise RuntimeError("算法执行错误")


def _partition(arr: List[Union[int, float]], left: int, right: int) -> int:
    """
    分区函数：将数组分为两部分。
    左侧元素 <= pivot，右侧元素 > pivot。
    
    使用随机选择 pivot 来避免最坏情况（已排序数组）。
    
    参数:
        arr: 待分区的数组
        left: 分区左边界
        right: 分区右边界
    
    返回:
        pivot 元素分区后的最终位置索引
    """
    # 随机选择 pivot 索引，避免最坏情况
    # 最坏情况发生在数组已排序且总是选择第一个或最后一个元素作为 pivot
    pivot_index = random.randint(left, right)
    
    # 将 pivot 交换到末尾
    _swap(arr, pivot_index, right)
    pivot = arr[right]
    
    # i 指向小于等于 pivot 的区域的最后一个元素
    i = left
    
    # 遍历当前区间
    for j in range(left, right):
        # 如果当前元素小于等于 pivot，将其放到左侧区域
        if arr[j] <= pivot:
            _swap(arr, i, j)
            i += 1
    
    # 将 pivot 放到正确的位置（i 位置）
    # 此时 i 左侧都 <= pivot，i 右侧都 > pivot
    _swap(arr, i, right)
    
    return i


def _swap(arr: List[Union[int, float]], i: int, j: int) -> None:
    """
    交换数组中两个元素的位置。
    
    参数:
        arr: 待操作的数组
        i: 第一个元素的索引
        j: 第二个元素的索引
    """
    arr[i], arr[j] = arr[j], arr[i]


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("快速选择算法 - 第 k 个最大数测试")
    print("=" * 50)
    
    # 测试用例 1：基本功能测试
    test1 = [3, 2, 1, 5, 6, 4]
    print(f"\n测试 1: 数组 = {test1}")
    for k in range(1, len(test1) + 1):
        result = quickselect_kth_largest(test1, k)
        print(f"  第 {k} 大的数: {result}")
    
    # 测试用例 2：重复元素
    test2 = [1, 3, 2, 3, 4, 3]
    print(f"\n测试 2: 数组 = {test2}")
    for k in range(1, len(test2) + 1):
        result = quickselect_kth_largest(test2, k)
        print(f"  第 {k} 大的数: {result}")
    
    # 测试用例 3：单个元素
    test3 = [42]
    print(f"\n测试 3: 数组 = {test3}")
    result = quickselect_kth_largest(test3, 1)
    print(f"  第 1 大的数: {result}")
    
    # 测试用例 4：边界情况 - k=1（最大值）
    test4 = [7, 2, 9, 1, 5, 8, 3]
    print(f"\n测试 4: 数组 = {test4}")
    result = quickselect_kth_largest(test4, 1)
    print(f"  第 1 大的数 (最大值): {result}")
    
    # 测试用例 5：边界情况 - k=n（最小值）
    result = quickselect_kth_largest(test4, len(test4))
    print(f"  第 {len(test4)} 大的数 (最小值): {result}")
    
    # 测试用例 6：错误处理 - 空数组
    print("\n测试 6: 错误处理 - 空数组")
    try:
        quickselect_kth_largest([], 1)
    except ValueError as e:
        print(f"  捕获异常: {e}")
    
    # 测试用例 7：错误处理 - k 超出范围
    print("\n测试 7: 错误处理 - k 超出范围")
    try:
        quickselect_kth_largest([1, 2, 3], 5)
    except ValueError as e:
        print(f"  捕获异常: {e}")
    
    # 测试用例 8：浮点数
    test8 = [3.5, 1.2, 4.8, 2.1, 5.9]
    print(f"\n测试 8: 数组 = {test8}")
    result = quickselect_kth_largest(test8, 2)
    print(f"  第 2 大的数: {result}")
    
    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)
