import redis
class Config(object):

    SECRET_KEY = "jdlakdjflkasdjflaskfla"

    SQLALCHEMY_DATABASE_URI = "mysql://root:mysql@127.0.0.1:3306/information11"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # SQLALCHEMY_COMMIT_ON_TEARDOWN =  True
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

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

# 动态的配置文件
config_map = {
    "development":DevelopmentConfig,
    "production":ProductionConfig
}