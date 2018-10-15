from flask import g
from flask import session

from info.models import User
import functools

def index_class(index):
    if index == 0:
        return "first"
    elif index == 1:
        return "second"
    elif index == 2:
        return "third"
    else:
        return ""

def user_login_data(f):
    @functools.wraps(f)
    def wrapper(*args,**kwargs):
        user_id = session.get("user_id")
        user = None
        if user_id:
            # 如果能够从session当中获取user_id,说明用户已经登陆
            user = User.query.get(user_id)
        g.user = user
        return f(*args,**kwargs)
    return wrapper


# def user_login_data():
#     # 从seesion当中获取到user_id,因为登陆成功之后,把user_id存储到session里面
#     user_id = session.get("user_id")
#     user = None
#     if user_id:
#         # 如果能够从session当中获取user_id,说明用户已经登陆
#         user = User.query.get(user_id)
#     return user