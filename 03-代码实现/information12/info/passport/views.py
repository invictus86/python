import random
import re
from datetime import datetime
from flask import request,jsonify,make_response,current_app
from flask import session

from info import constants, db
from info import redis_store
from info.models import User
from info.utils.captcha.captcha import captcha
from info.utils.response_code import RET
from info.utils.yuntongxun.sms import CCP
from . import passport_blue

@passport_blue.route("/logout")
def logout():
    # 清除session里面的数据
    session.pop("user_id",None)
    session.pop("mobile",None)
    session.pop("nick_name",None)
    session.pop("is_admin",None)
    return jsonify(errno = RET.OK,errmsg = "退出成功")


# ctrl + shift + t
@passport_blue.route("/login",methods = ["POST"])
def login():
    mobile = request.json.get("mobile")
    password = request.json.get("password")
    # 从数据库查询user对象
    user = User.query.filter(User.mobile == mobile).first()

    if not user:
        return jsonify(errno = RET.NODATA,errmsg = "用户不存在")

    if not user.check_password(password):
        return jsonify(errno = RET.PARAMERR,errmsg = "请输入正确的密码")
    # 把user数据存储到session里面,然后用来保持状态链接
    session["user_id"] = user.id
    session["mobile"] = user.mobile
    session["nick_name"] = user.nick_name
    # 获取到当前时间
    user.last_login = datetime.now()
    # 更新操作,不需要add
    # db.session.commit()
    return jsonify(errno = RET.OK,errmsg = "登陆成功")








@passport_blue.route("/register",methods = ["POST"])
def register():
    mobile = request.json.get("mobile")
    smscode = request.json.get("smscode")
    password = request.json.get("password")

    if not all([mobile,smscode,password]):
        return jsonify(errno = RET.PARAMERR,errmsg = "参数输入错误")

    # 校验手机验证码
    # 从redis里面获取刚刚存储的短信验证码
    redis_sms_code = redis_store.get("code_" + mobile)

    if not redis_sms_code:
        return jsonify(errno = RET.NODATA,errmsg = "验证码已经过期")
    # 校验服务端的短信验证码和用户输入的验证码是否一致
    if redis_sms_code != smscode:
        return jsonify(errno = RET.PARAMERR,errmsg = "验证码输入错误")


    user = User()
    user.mobile = mobile
    user.nick_name = mobile
    user.password = password
    # 获取到当前时间
    user.last_login = datetime.now()

    # 把user对象添加到数据库
    try:
        db.session.add(user)
        db.session.commit()
    except Exception as e:
        current_app.logger.error(e)
        db.session.rollback()

    session["user_id"] = user.id
    session["nick_name"] = user.nick_name
    session["mobile"] = user.mobile
    return jsonify(errno = RET.OK,errmsg = "注册成功")






@passport_blue.route("/sms_code",methods = ["POST"])
def sms_code():
    # 获取到用户在页面上输入的手机号码
    mobile = request.json.get("mobile")
    # 获取到用户在页面上输入的验证码
    image_code = request.json.get("image_code")
    # 获取到验证码的id
    image_code_id = request.json.get("image_code_id")

    if not all([mobile,image_code,image_code_id]):
        return jsonify(errno = RET.PARAMERR,errmsg = "请输入参数")

    if not re.match("1[345678]\d{9}",mobile):
        return jsonify(errno = RET.PARAMERR,errmsg = "请输入正确的手机号")

    # 从redis当中获取到图片验证码
    redis_image_code = redis_store.get("sms_code_"+ image_code_id)
    # 判断验证码是否过期
    if not redis_image_code:
        return jsonify(errno = RET.NODATA,errmsg = "验证码已过期")

    # 在匹配验证码的时候,需要全部转换成大写或者小写,这样是为了提高用户体验
    if redis_image_code.lower() != image_code.lower():
        return jsonify(errno = RET.PARAMERR,errrmsg = "请输入正确的验证码")

    # 发送短信
    # 生成6位数字的验证码
    result = random.randint(0,999999)
    # 保证每次获取到的验证码肯定是6位
    sms_code = "%06d"%result
    print("短信验证码 = " + sms_code)
    # 存储短信验证码到redis当中
    # 第一个参数表示redis的key
     # 第二个参数表示redis的值
    #第三个参数表示过期时间,单位是秒
    redis_store.set("code_" + mobile,sms_code,300)

    # 发送短信验证码
    # 第一个参数表示你想发送给哪个手机号
    # 第二个参数是一个数组[sms_code,10],数组里面的第一个参数表示随机验证码,第二个参数表示过期时间,单位是分钟
    # statusCode = CCP().send_template_sms(mobile,[sms_code,10],1)
    #
    # if statusCode != 0:
    #     return jsonify(errno = RET.THIRDERR,errmsg = "发送短信失败")

    return jsonify(errno = RET.OK,errmsg = "短信发送成功")










# 双击shift
@passport_blue.route("/image_code")
def image_code():
    print("前端请求的url地址 = " + request.url)
    # 获取到前端页面传递过来的id
    code_id = request.args.get("code_id")

    if not code_id:
        return jsonify(errno = RET.PARAMERR,errmsg = "参数错误")

    # 获取图片验证码:
    # name:图片验证码的名字
    # text:图片验证码的内容
    # image:图片验证码
    name, text, image = captcha.generate_captcha()
    print("图片验证码 = " + text)
    # 把uui存储到redis当中
    # 第一个参数表示key
    # 第二个参数表示value
    # 第三个参数表示过期时间,单位是秒
    redis_store.set("sms_code_"+ code_id,text,300)
    # response接受一个响应体
    resp = make_response(image)
    resp.headers['Content-Type'] = 'image/jpg'
    return resp
