from sys import argv
import argparse
# argv包用来显示当前python文件的配置参数，这些参数通常是在用命令行运行文件时才有用

parser = argparse.ArgumentParser(description="the default config of progress") # 创建配置对象
parser.add_argument("--a", type=int, default=2, help="the first variable")    # 添加配置
parser.add_argument("--b", type=int, default=1, help="the second variable")
arg = parser.parse_args()    # 解析配置并传入arg变量，可以方便我们在程序任意位置调用


def func(a, b):
    print(a + b)

if __name__ == "__main__":
    func(arg.a, arg.b)    # 通过配置传参
    # print(arg)   打印参数

    # 当我们拿到一个别人的项目，不知道有哪些可配置的参数时就可以使用python 文件名.py -h查看信息
    # 这样就可以很清楚的知道项目如何设置参数运行