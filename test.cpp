#include <iostream>
#include <unordered_map>
#include <vector>

std::vector<int> twoSum(
    const std::vector<int>& nums,
    int target
) {
    std::unordered_map<int, int> visited;

    for (int i = 0; i < nums.size(); ++i) {
        int need = target - nums[i];

        if (visited.contains(need)) {
            return {visited[need], i};
        }

        visited[nums[i]] = i;
    }

    return {};
}

int main() {
    std::vector<int> nums = {2, 7, 11, 15};
    int target = 9;

    std::vector<int> result = twoSum(nums, target);

    if (result.size() == 2) {
        std::cout << "下标为："
                  << result[0] << " "
                  << result[1] << '\n';

        std::cout << "对应数字为："
                  << nums[result[0]] << " "
                  << nums[result[1]] << '\n';
    } else {
        std::cout << "没有找到符合条件的两个数\n";
    }

    return 0;
}