from flask_migrate import Migrate,MigrateCommand
from flask_script import Manager

from info import create_app,db
from info import models
from info.models import User

"""
manager:只是负责启动当前应用程序


"""
app = create_app("development")
manager = Manager(app)
Migrate(app,db)
manager.add_command("mysql",MigrateCommand)
"""
添加一个管理员到数据库里面
"""
@manager.option('-n', '--name', dest='name')
@manager.option('-p', '--password', dest='password')
def create_super_user(name,password):
    # 初始化管理员的数据
    user = User()
    user.nick_name = name
    user.mobile = name
    user.is_admin = True
    user.password = password
    # 提交到数据
    db.session.add(user)
    db.session.commit()









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




if __name__ == '__main__':
    print(app.url_map)
    manager.run()