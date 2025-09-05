import matplotlib.pyplot as plt
import json

# 你的Python代码逻辑（替换成你自己的计算）
data = {
    "name": "Python分析结果",
    "values": [25, 40, 30, 50, 20]
}

# 生成图表
plt.bar(range(len(data["values"])), data["values"])
plt.title("数据分析结果")
plt.savefig("result.png")  # 保存图片

# 保存数据
with open("data.json", "w") as f:
    json.dump(data, f)

print("已生成 result.png 和 data.json")