import functools
# call user_login_data
# call wrapper
# call num1
# aaa

def user_login_data(f):
    # print("call user_login_data")
    @functools.wraps(f)
    def wrapper(*args,**kwargs):
        # 实现我们的业务逻辑

        # print("call wrapper")
        return f(*args,**kwargs)
    return wrapper

@user_login_data
def num1():
    # print("call num1")
    print("aaa")

@user_login_data
def num2():
    # print("call num1")
    print("aaa")

if __name__ == '__main__':
    print(num1.__name__)
    print(num2.__name__)
   # num1()