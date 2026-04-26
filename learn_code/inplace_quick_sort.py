def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # 分区操作，返回基准元素的最终位置
        pi = partition(arr, low, high)
        
        # 递归排序基准元素左侧和右侧
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)


def partition(arr, low, high):
    # 选择最后一个元素作为基准
    pivot = arr[high]
    
    # i 是小于基准的元素的索引
    i = low - 1
    
    # 遍历数组，将小于基准的元素移到左边
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # 将基准元素放到正确位置
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# 测试函数
if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("原数组:", test_array)
    quick_sort(test_array)
    print("排序后数组:", test_array)