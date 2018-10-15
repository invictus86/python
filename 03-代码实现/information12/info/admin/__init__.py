from flask import Blueprint
from flask import request
from flask import session,redirect

admin_blue = Blueprint("admin",__name__,url_prefix="/admin")

from . import views
"""
权限校验，普通用户只能登陆前台页面，不能登陆后台管理员的界面
"""
# 请求函数之前调用
@admin_blue.before_request
def check_admin():
    is_admin = session.get("is_admin",False)
    if not is_admin and not request.url.endswith("/admin/login"):
        return redirect("/")
