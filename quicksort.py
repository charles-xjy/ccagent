"""
原地快速排序算法 (In-place Quicksort)

快速排序是一种分治排序算法，基本思想是：
1. 选择一个基准元素 (pivot)
2. 将数组分区，小于基准的元素放在左边，大于基准的放在右边
3. 递归地对左右子数组进行排序

时间复杂度：平均 O(n log n)，最坏 O(n²)
空间复杂度：O(log n) - 递归调用栈空间
"""


def partition(arr, low, high):
    """
    分区函数：将数组分为两部分
    - 小于基准值的元素在左边
    - 大于等于基准值的元素在右边
    
    参数:
        arr: 待排序的数组
        low: 子数组的起始索引
        high: 子数组的结束索引
    
    返回:
        基准元素的最终位置索引
    """
    # 选择最右边的元素作为基准值 (pivot)
    pivot = arr[high]
    
    # i 指向小于 pivot 区域的最后一个元素
    i = low - 1
    
    # 遍历数组，将小于 pivot 的元素移到左边
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            # 交换 arr[i] 和 arr[j]
            arr[i], arr[j] = arr[j], arr[i]
    
    # 将基准值放到正确的位置（i+1）
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    
    return i + 1


def quicksort(arr, low, high):
    """
    快速排序主函数（递归实现）
    
    参数:
        arr: 待排序的数组
        low: 子数组的起始索引
        high: 子数组的结束索引
    """
    # 递归终止条件：子数组长度为 0 或 1
    if low < high:
        # 分区操作，获取基准元素的正确位置
        pivot_index = partition(arr, low, high)
        
        # 递归排序基准元素左边的子数组
        quicksort(arr, low, pivot_index - 1)
        
        # 递归排序基准元素右边的子数组
        quicksort(arr, pivot_index + 1, high)


def quicksort_wrapper(arr):
    """
    快速排序的包装函数，提供简洁的调用接口
    
    参数:
        arr: 待排序的数组
    
    返回:
        排序后的数组（原地修改，返回同一个数组对象）
    """
    if len(arr) <= 1:
        return arr
    
    quicksort(arr, 0, len(arr) - 1)
    return arr


def test_quicksort():
    """测试快速排序算法的正确性"""
    
    # 测试用例 1：普通数组
    arr1 = [64, 34, 25, 12, 22, 11, 90]
    quicksort_wrapper(arr1)
    expected1 = [11, 12, 22, 25, 34, 64, 90]
    assert arr1 == expected1, f"测试 1 失败：{arr1} != {expected1}"
    print("✓ 测试 1 通过：普通数组排序")
    
    # 测试用例 2：已排序数组
    arr2 = [1, 2, 3, 4, 5]
    quicksort_wrapper(arr2)
    expected2 = [1, 2, 3, 4, 5]
    assert arr2 == expected2, f"测试 2 失败：{arr2} != {expected2}"
    print("✓ 测试 2 通过：已排序数组")
    
    # 测试用例 3：逆序数组
    arr3 = [5, 4, 3, 2, 1]
    quicksort_wrapper(arr3)
    expected3 = [1, 2, 3, 4, 5]
    assert arr3 == expected3, f"测试 3 失败：{arr3} != {expected3}"
    print("✓ 测试 3 通过：逆序数组")
    
    # 测试用例 4：包含重复元素
    arr4 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    quicksort_wrapper(arr4)
    expected4 = [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
    assert arr4 == expected4, f"测试 4 失败：{arr4} != {expected4}"
    print("✓ 测试 4 通过：包含重复元素")
    
    # 测试用例 5：单元素数组
    arr5 = [42]
    quicksort_wrapper(arr5)
    expected5 = [42]
    assert arr5 == expected5, f"测试 5 失败：{arr5} != {expected5}"
    print("✓ 测试 5 通过：单元素数组")
    
    # 测试用例 6：空数组
    arr6 = []
    quicksort_wrapper(arr6)
    expected6 = []
    assert arr6 == expected6, f"测试 6 失败：{arr6} != {expected6}"
    print("✓ 测试 6 通过：空数组")
    
    # 测试用例 7：包含负数
    arr7 = [-5, 3, -1, 0, 2, -8, 7]
    quicksort_wrapper(arr7)
    expected7 = [-8, -5, -1, 0, 2, 3, 7]
    assert arr7 == expected7, f"测试 7 失败：{arr7} != {expected7}"
    print("✓ 测试 7 通过：包含负数")
    
    # 测试用例 8：验证原地排序（不创建新数组）
    arr8 = [3, 1, 2]
    original_id = id(arr8)
    result = quicksort_wrapper(arr8)
    assert id(result) == original_id, "测试 8 失败：不是原地排序"
    assert arr8 == [1, 2, 3], "测试 8 失败：排序结果不正确"
    print("✓ 测试 8 通过：原地排序验证")
    
    print("\n" + "=" * 40)
    print("所有测试用例通过！✓")
    print("=" * 40)


if __name__ == "__main__":
    test_quicksort()
    
    # 演示示例
    print("\n示例演示:")
    example = [10, 7, 8, 9, 1, 5]
    print(f"排序前：{example}")
    quicksort_wrapper(example)
    print(f"排序后：{example}")
