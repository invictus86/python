from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from flask_wtf.csrf import CSRFProtect
import redis
from flask_script import Manager
from flask_migrate import Migrate,MigrateCommand

"""
manager:只是负责启动当前应用程序


"""



app = Flask(__name__)

class Config(object):
    DEBUG = True
    SECRET_KEY = "jdlakdjflkasdjflaskfla"

    SQLALCHEMY_DATABASE_URI = "mysql://root:mysql@127.0.0.1:3306/information12"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # ip地址
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379


    # 如果设置flask-session,如下三种配置是固定的写法
    # 设置数据库的类型
    SESSION_TYPE = "redis"
    # 表示session两天之后过期
    PERMANENT_SESSION_LIFETIME = 86400 * 2
    # 开启session签名
    SESSION_USE_SIGNER = True
    # 把flasksession放置到redis数据库当中
    SESSION_REDIS = redis.StrictRedis(host=REDIS_HOST,port=REDIS_PORT)


app.config.from_object(Config)
# 数据库需要在app加载config文件之后执行,不然会直接报一个警告
db = SQLAlchemy(app)
# 初始化redis的存储数据对象(短信验证码,图片验证码)
redis_store = redis.StrictRedis(host=Config.REDIS_HOST,port=Config.REDIS_PORT,decode_responses=True)



Session(app)
# 开启CSRF保护
CSRFProtect(app)

manager = Manager(app)
Migrate(app,db)
manager.add_command("mysql",MigrateCommand)


"""
通用的配置:
1 数据库
2 开启csrf保护
3 session
4 redis
5 数据库迁移
6 日志
7 蓝图



"""


@app.route("/")
def index():
    return "index page"

if __name__ == '__main__':
    manager.run()