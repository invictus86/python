import logging
from logging.handlers import RotatingFileHandler

from flask import Flask,render_template
from flask import g

from config import config_map, DevelopmentConfig, ProductionConfig
from flask_sqlalchemy import SQLAlchemy
import redis
from flask_session import Session
from flask_wtf.csrf import CSRFProtect
from flask_wtf.csrf import generate_csrf



# 如下设置日志是固定的写法(大家直接拷贝)
# 设置日志的记录等级
logging.basicConfig(level=logging.DEBUG) # 调试debug级
# 创建日志记录器，指明日志保存的路径、每个日志文件的最大大小、保存的日志文件个数上限
file_log_handler = RotatingFileHandler("logs/log", maxBytes=1024 * 1024 * 100, backupCount=10)
# 创建日志记录的格式 日志等级 输入日志信息的文件名 行数 日志信息
formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)d %(message)s')
# 为刚创建的日志记录器设置日志记录格式
file_log_handler.setFormatter(formatter)
# 为全局的日志工具对象（flask app使用的）添加日志记录器
logging.getLogger().addHandler(file_log_handler)


redis_store = None # type:redis.StrictRedis

db = SQLAlchemy()
# 创建并且返回flask对象
def create_app(config_name):
    app = Flask(__name__)
    # 在不动当前代码的同时,修改config类的名字
    config_class = config_map.get(config_name)
    app.config.from_object(config_class)
    # 数据库需要在app加载config文件之后执行,不然会直接报一个警告
    db.init_app(app)
    global redis_store
    # 初始化redis的存储数据对象(短信验证码,图片验证码)
    redis_store = redis.StrictRedis(host=config_class.REDIS_HOST,port=config_class.REDIS_PORT,decode_responses=True)

    Session(app)
    # 开启CSRF保护
    CSRFProtect(app)
    # 通过请求钩子,往cookie里面保存token
    @app.after_request
    def after_request(response):
        # 生成一个csrf_token
        csrf_token = generate_csrf()
        response.set_cookie("csrf_token",csrf_token)
        return response

    from info.utils.common import index_class
    # 添加过滤器
    app.add_template_filter(index_class,"indexClass")

    from info.utils.common import user_login_data
    # 捕捉全局的404异常
    @app.errorhandler(404)
    @user_login_data
    def not_found(error):
        user = g.user
        data = {
            "user_info":user.to_dict() if user else None
        }
        return render_template("news/404.html",data = data)


    # 在需要导入模块的时候,才import
    from info.index import index_blue
    app.register_blueprint(index_blue)

    # 在需要导入模块的时候,才import
    from info.passport import passport_blue
    app.register_blueprint(passport_blue)

    # 在需要导入模块的时候,才import
    from info.news import news_blue
    app.register_blueprint(news_blue)

    # 在需要导入模块的时候,才import
    from info.user import profile_blue
    app.register_blueprint(profile_blue)

    # 在需要导入模块的时候,才import
    from info.admin import admin_blue
    app.register_blueprint(admin_blue)

    return app