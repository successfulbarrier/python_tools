# 求两个正序数列加一起的中位数
from typing import List


def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    sum = 0
    len_num1 = len(nums1)
    len_num2 = len(nums2)
    len_total = len_num1 + len_num2
    len_total_n = (len_num1 + len_num2) / 2
    x = 0
    y = 0
    value = 0
    print(int(len_total_n))

    # 判断是否有数组为空
    if len_num1 == 0 and len_num2 == 0:
        print("请不要输入空数组！！！！！")
    elif len_num2 == 0 :
        if len_num1 % 2 == 1 :
            value = nums1[int(len_num1/2)]
            return value
        elif len_num1 % 2 == 0 :
            value = (nums1[int(len_num1/2)] + nums1[int(len_num1/2 - 1)])/2
            return value

    elif len_num1 == 0 :
        if len_num2 % 2 == 1 :
            value = nums2[int(len_num2/2)]
            return value
        elif len_num2 % 2 == 0 :
            value = (nums2[int(len_num2/2)] + nums2[int(len_num2/2 - 1)])/2
            return value

    if len_total % 2 == 1:
        for i in range(int(len_total_n)):
            if nums1[x] <= nums2[y]:
                x = x + 1

            elif nums1[x] > nums2[y]:
                y = y + 1

        if nums1[x] < nums2[y]:
            value = nums1[x]
        else:
            value = nums2[y]

    elif len_total % 2 == 0:
        for i in range(int(len_total_n-1)):
            if nums1[x] <= nums2[y]:
                x = x + 1

            elif nums1[x] > nums2[y]:
                y = y + 1

        value = (nums1[x] + nums2[y]) / 2

    return value


def test01():
    nums1 = [1, 3, 5, 7, 10, 12]
    nums2 = []
    print(findMedianSortedArrays(nums1, nums2))


def test02():
    nums1 = [1, 2]
    nums2 = [-1, 3]
    print(findMedianSortedArrays(nums1, nums2))


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    test01()
    test02()