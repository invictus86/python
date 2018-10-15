from flask import current_app
from flask import g
from flask import request
from flask import session

from info import constants
from info import db
from info.models import Category, News, User
from info.utils.common import user_login_data
from info.utils.response_code import RET
from . import profile_blue
from flask import render_template,redirect,jsonify
from info.utils.image_storage import storage
'''
其他用户的详情页面
'''
@profile_blue.route("/other_info")
@user_login_data
def other_info():
    # 我关注的那个用户的id
    id = request.args.get("id")
    other = User.query.get(id)
    user = g.user
    # 当前登陆用户没有关注新闻的作者,默认情况下,肯定是没有关注,所以设置为false
    is_followed = False
    # 如果当前登陆用户需要关注新闻作者的话,那么必须首先得有新闻作者
    # 用户登陆之后,才能进行关注
    if other and user:
        # 如果当前新闻的作者,在登陆用户的关注人列表当中,就可以说明当前登陆作者是新闻作者的粉丝
        if other in user.followed:
            is_followed = True


    data = {
        "other_info":other.to_dict(),
        "user_info":user.to_dict() if user else None,
        "is_followed": is_followed
    }

    return render_template("news/other.html",data = data)





"""
我的关注
"""
@profile_blue.route("/user_follow")
@user_login_data
def follow():
    page = request.args.get("p",1)
    try:
        page = int(page)
    except Exception as e:
        current_app.logger.error(e)
        page = 1
    user = g.user
    # 查询我所有关注的用户
    paginate = user.followed.paginate(page,4,False)
    items = paginate.items
    current_page = paginate.page
    total_page = paginate.pages
    users = []
    for item in items:
        users.append(item.to_dict())
    data = {
        "users":users,
        "current_page":current_page,
        "total_page":total_page
    }

    return render_template("news/user_follow.html",data = data)





"""
作者发布的新闻列表
"""
@profile_blue.route("/news_list")
@user_login_data
def news_list():
    page = request.args.get("p",1)
    try:
        page = int(page)
    except Exception as e:
        current_app.logger.error(e)

    user = g.user
    # 查询我自己发布的新闻
    paginate = news = News.query.filter(News.user_id == user.id).paginate(page,10,False)
    items = paginate.items
    current_page = paginate.page
    total_page  = paginate.pages
    news_list = []
    for item in items:
        news_list.append(item.to_review_dict())

    data = {
        "news_list":news_list,
        "current_page":current_page,
        "total_page":total_page
    }

    return render_template("news/user_news_list.html",data = data)




"""
作者发布新闻
"""
@profile_blue.route("/news_release",methods = ["GET","POST"])
@user_login_data
def news_release():
    if request.method == "GET":
        category_list = Category.query.all()
        categories = []
        for category in category_list:
            categories.append(category.to_dict())
        # 最新的分类是按照时间进行排序，所有不需要添加到分类列表当中
        categories.pop(0)
        data = {
            "categories":categories
        }
        return render_template("news/user_news_release.html",data = data)

    title = request.form.get("title")
    category_id = request.form.get("category_id")
    digest = request.form.get("digest")
    index_image = request.files.get("index_image")
    content = request.form.get("content")
    if not all([title,category_id,digest,index_image,content]):
        return jsonify(errno = RET.PARAMERR,errmsg = "参数输入错误")

    index_image = index_image.read()
    key = storage(index_image)
    user = g.user
    news = News()
    news.title = title
    news.source = "个人来源"
    news.digest = digest
    news.content = content
    news.index_image_url = constants.QINIU_DOMIN_PREFIX + key
    news.category_id = category_id
    news.user_id = user.id
    # 1表示当前新闻在审核中
    news.status = 1
    db.session.add(news)
    db.session.commit()
    return jsonify(errno = RET.OK,errmsg = "发布成功")

"""
用户收藏
"""
@profile_blue.route("/collection")
@user_login_data
def collection():
    page = request.args.get("p",1)
    try:
        page = int(page)
    except Exception as e:
        current_app.logger.error(e)
        page = 1
    user = g.user
    paginate = user.collection_news.paginate(page,2,False)
    items = paginate.items
    current_page = paginate.page
    total_page = paginate.pages
    collections = []
    for item in items:
        collections.append(item.to_dict())
    data = {
        "collections":collections,
        "current_page":current_page,
        "total_page":total_page,
    }
    return render_template("news/user_collection.html",data = data)



"""
密码修改
"""
@profile_blue.route("/pass_info",methods = ["GET","POST"])
@user_login_data
def pass_info():
    if request.method == "GET":
       return render_template("news/user_pass_info.html")
    old_password = request.json.get("old_password")
    new_password = request.json.get("new_password")
    if not all([old_password,new_password]):
        return jsonify(errno = RET.PARAMERR,errmsg = "请输入密码")
    user = g.user
    if not user.check_password(old_password):
        return jsonify(errno = RET.PARAMERR,errmsg = "密码输入错误")

    user.password = new_password
    db.session.commit()
    return jsonify(errno = RET.OK,errmsg = "修改成功")




@profile_blue.route("/pic_info",methods = ["GET","POST"])
@user_login_data
def pic_info():
    user = g.user

    if request.method == "GET":
        data = {
            "user_info": user.to_dict() if user else None
        }
        return render_template("news/user_pic_info.html", data=data)

    avatar_file = request.files.get("avatar").read()

    key = storage(avatar_file)
    # 设置头像
    user.avatar_url = key
    db.session.commit()

    data = {
        "avatar_url" : constants.QINIU_DOMIN_PREFIX + key
    }

    return jsonify(errno = RET.OK,errmsg = "上传成功",data = data )






@profile_blue.route("/base_info",methods = ["GET","POST"])
@user_login_data
def base_info():
    user = g.user

    if request.method == "GET":
        data = {
            "user_info": user.to_dict() if user else None
        }
        return render_template("news/user_base_info.html",data = data)

    nick_name =  request.json.get("nick_name")
    signature =  request.json.get("signature")
    gender =  request.json.get("gender")

    user.nick_name  = nick_name
    user.signature = signature
    user.gender = gender

    db.session.commit()

    session["nick_name"] = user.nick_name

    return jsonify(errno = RET.OK,errmsg = "修改成功")








@profile_blue.route("/info")
@user_login_data
def info():
    user = g.user
    if not user:
        return redirect("/")
    data = {
        "user_info": user.to_dict() if user else None
    }
    return render_template("news/user.html",data = data)
