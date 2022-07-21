import os

print("当前目录：", os.getcwd())   # 得到当前目录地址

# 返回指定目录下的所有文件和目录名
root_dir = os.listdir("C:/Users/EVER/Desktop/Python-DL_basic/")
print(root_dir)
origin_path = "C:/Users/EVER/Desktop/Python-DL_basic/"
os.makedirs(origin_path + 'test', exist_ok=True)   # 在指定目录下生成一个新的文件夹（目录）

path = os.path.join(origin_path + "test/")    # 组合目录名生成新的path字符串
print(path)

# 返回一个元组。元组第一个元素为文件所在目录，第二个元素为文件名（含扩展名）
print(os.path.split(path + "test.txt")[0])
print(os.path.split(path + "test.txt")[1])


with open(path + "test.txt", "w") as f:    # 在path目录下创建一个test文本文件并写入内容
    txt = ["hello", "world"]
    f.write(txt[0])
    f.write("\n")
    f.write(txt[1])

with open(path + "test.txt", "r") as f:
    for item in f.readlines():
        print(item.split("\n")[0])     # 去掉换行符输出
    
    # 另一种办法
    # print(f.read().splitlines())