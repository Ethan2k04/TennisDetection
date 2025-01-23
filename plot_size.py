import matplotlib.pyplot as plt


# 文件路径
SIZE_FILE = "size_data.txt"


def read_size_data():
    """
    从文件中读取size数据
    """
    sizes = []
    with open(SIZE_FILE, "r") as file:
        for line in file:
            # 将每行的字符串转换为整数列表
            size = list(map(int, line.strip().split(",")))
            sizes.append(size)
    return sizes


def plot_size_data(sizes):
    """
    绘制size数据的变化折线图
    """
    plt.figure(figsize=(10, 6))
    for i, size in enumerate(sizes):
        plt.plot(size, label=f"Size Array {i+1}")  # 绘制每条size数组的折线
    plt.xlabel("Index")
    plt.ylabel("Size Value")
    plt.title("Size Array Changes Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 读取size数据
    sizes = read_size_data()
    # 绘制折线图
    plot_size_data(sizes)
