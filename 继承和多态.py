### 面向对象编程语言特有属性，多态是基于继承关系的
import argparse

## python语言自带多态功能
## 什么是多态？（类比函数重载，只是在类方法中称为多态）
## 从一个父类派生出多个子类，可以使子类之间有不同的行为，这种行为称之为多态
## 更直白的说，就是子类重写父类的方法，使子类具有不同的方法实现
## 子类与父类拥有同一个方法，子类的方法优先级高于父类，即子类覆盖父类

class Person():
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender
    def whoAmI(self):
        print( 'I am a Person, my name is %s' % self.name)

class Student(Person):    # 继承Person类
    def __init__(self, name, gender, score):    # 重写init方法，实例化子类时，就不会自动调用父类的构造函数
        super(Student, self).__init__(name, gender)   # 必须用super方法强制调用
        self.score = score
    def whoAmI(self):
        print( 'I am a Student, my score is %d' % self.score)   # 重写类方法即多态

def who_am_i(person):    # 函数包装一下：调用类方法
    person.whoAmI()

if __name__ == "__main__":
    p = Person("bob", "male")
    s = Student("mary", "female", 100)
    who_am_i(p)
    who_am_i(s)