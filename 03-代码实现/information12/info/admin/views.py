from flask import current_app
from flask import g
from flask import session, jsonify

from info import constants
from info.utils.image_storage import storage
from info import db
from info.utils.common import user_login_data
from info.models import User, News, Category
from info.utils.response_code import RET
from . import admin_blue
from flask import render_template, request, redirect, url_for
import time
from datetime import datetime, timedelta
@admin_blue.route('/add_category', methods=["POST"])
def add_category():
    """修改或者添加分类"""
    category_id = request.json.get("id")
    name = request.json.get("name")


    if category_id:
        category = Category.query.get(category_id)
        category.name = name
    else:
        category = Category()
        category.name = name

        db.session.add(category)

    db.session.commit()

    return jsonify(errno = RET.OK,errmsg = "OK")





@admin_blue.route("/news_type")
def news_type():
    category_list = Category.query.all()
    categories = []
    for category in category_list:
        categories.append(category.to_dict())
    categories.pop(0)
    data = {
        "categories":categories
    }
    return render_template("admin/news_type.html",data = data)



@admin_blue.route("/news_edit_detail", methods=["GET", "POST"])
def news_edit_detail():
    if request.method == "GET":
        news_id = request.args.get("news_id")
        news = News.query.get(news_id)
        category_list = Category.query.all()
        categories = []
        for category in category_list:
            categories.append(category.to_dict())
        categories.pop(0)
        data = {
            "news": news.to_dict(),
            "categories": categories
        }

        return render_template("admin/news_edit_detail.html", data=data)

    news_id = request.form.get("news_id")
    
    news = News.query.get(news_id)
    
    
    title = request.form.get("title")
    digest = request.form.get("digest")
    content = request.form.get("content")
    index_image = request.files.get("index_image")
    category_id = request.form.get("category_id")
    # 1.1 判断数据是否有值
    if not all([title, digest, content, index_image, category_id]):
        return jsonify(errno=RET.PARAMERR, errmsg="参数有误")

    index_image = index_image.read()
    key = storage(index_image)

    news.title = title
    news.digest = digest
    news.content = content
    news.index_image_url = constants.QINIU_DOMIN_PREFIX + key
    news.category_id = category_id

    db.session.commit()
    return jsonify(errno = RET.OK,errmsg  = "编辑成功")
    


"""
新闻编辑页面

"""


@admin_blue.route("/news_edit")
def news_edit():
    page = request.args.get("p", 1)
    # 获取到小编搜索的文字
    keywords = request.args.get("keywords")
    try:
        page = int(page)
    except Exception as e:
        current_app.logger.error(e)
        page = 1

    paginate = News.query.order_by(News.create_time.desc()).paginate(page, 10, False)
    items = paginate.items
    current_page = paginate.page
    total_page = paginate.pages

    news_list = []
    for item in items:
        news_list.append(item.to_review_dict())

    data = {
        "news_list": news_list,
        "current_page": current_page,
        "total_page": total_page
    }
    return render_template("admin/news_edit.html", data=data)


"""
新闻审核的详情页面
"""


@admin_blue.route("/news_review_detail", methods=["GET", "POST"])
def news_review_detail():
    if request.method == "GET":
        news_id = request.args.get("news_id")
        news = News.query.get(news_id)
        data = {
            "news": news.to_dict()
        }
        return render_template("admin/news_review_detail.html", data=data)

    action = request.json.get("action")
    news_id = request.json.get("news_id")

    news = News.query.get(news_id)

    if action == "accept":
        # 审核通过,如果审核通过,那么直接修改当前新闻的状态就可以了
        news.status = 0
    else:
        # 审核不通过,拒绝,拒绝就需要说明原因
        reason = request.json.get("reason")
        if not reason:
            return jsonify(errno=RET.NODATA, errmsg="请说明拒绝原因")

        news.status = -1
        news.reason = reason

    db.session.commit()
    return jsonify(errno=RET.OK, errmsng="ok")


"""
新闻审核
"""


@admin_blue.route("/news_review")
def news_review():
    page = request.args.get("p", 1)
    # 获取到小编搜索的文字
    keywords = request.args.get("keywords")
    try:
        page = int(page)
    except Exception as e:
        current_app.logger.error(e)
        page = 1

    filters = [News.status != 0]
    # 如果小编用了搜索功能,才需要进行搜索,如果不需要搜索,那么就不需要查询数据库
    if keywords:
        filters.append(News.title.contains(keywords))
    paginate = News.query.filter(*filters).order_by(News.create_time.desc()).paginate(page, 10, False)
    items = paginate.items
    current_page = paginate.page
    total_page = paginate.pages

    news_list = []
    for item in items:
        news_list.append(item.to_review_dict())

    data = {
        "news_list": news_list,
        "current_page": current_page,
        "total_page": total_page
    }

    return render_template("admin/news_review.html", data=data)


"""
获取到用户列表
"""


@admin_blue.route("/user_list")
def user_list():
    page = request.args.get("p", 1)
    try:
        page = int(page)
    except Exception as e:
        current_app.logger.error(e)
        page = 1
    paginate = User.query.filter(User.is_admin == False).order_by(User.last_login.desc()).paginate(page, 10, False)
    items = paginate.items
    current_page = paginate.page
    total_page = paginate.pages

    users = []
    for user in items:
        users.append(user.to_admin_dict())

    data = {
        "users": users,
        "current_page": current_page,
        "total_page": total_page
    }

    return render_template("admin/user_list.html", data=data)


"""
统计当前数据库里面的用户数量
"""


@admin_blue.route("/user_count")
def user_count():
    # 用户总数量
    total_count = 0
    # 一个月新增加的用户数量
    mon_count = 0
    # 一天新增加的用户数量
    day_count = 0

    sum_count = User.query.filter(User.is_admin == False).count()

    t = time.localtime()
    # 2018-06-01
    mon_begin = "%d-%02d-01" % (t.tm_year, t.tm_mon)
    # 第一个参数表示时间字符串,
    # 第二个参数表示格式化时间
    mon_begin_date = datetime.strptime(mon_begin, "%Y-%m-%d")

    # 查询本月新增加了多少用户量
    mon_count = User.query.filter(User.is_admin == False, User.create_time > mon_begin_date).count()

    # 2018-06-01
    day_begin = "%d-%02d-%02d" % (t.tm_year, t.tm_mon, t.tm_mday)
    # 第一个参数表示时间字符串,
    # 第二个参数表示格式化时间
    day_begin_date = datetime.strptime(day_begin, "%Y-%m-%d")

    # 查询本月新增加了多少用户量
    day_count = User.query.filter(User.is_admin == False, User.create_time > day_begin_date).count()

    # 查询今天开始的时间


    today_begin = "%d-%02d-%02d" % (t.tm_year, t.tm_mon, t.tm_mday)
    # 第一个参数表示时间字符串,
    # 第二个参数表示格式化时间
    # 2018-06-18 0点0分0秒
    today_begin_date = datetime.strptime(day_begin, "%Y-%m-%d")
    activate_count = []
    activate_time = []

    # 需求:查询过去一个月里面所有的活跃用户
    # 我们需要查询的是用户,admin不是用户,是员工
    # User.query.filter(User.is_admin == False,User.create_time > 5月20号0点0分0秒 and
    #                                         User.create_time < 今天 的23点59分59秒)

    # t = time.localtime()
    # 如果需要算过去一个月的用户活跃量
    # 需要先把今天一天的活跃用户算出来
    # for i in range(0,31):
    #     # today_date = 2018-06月21 0点0分0秒
    #
    #     today_date = today_begin_date - timedelta(days= i)
    #     # end_day_date = 6月22 0点0分0秒(6月21号 23点59分59秒)
    #
    #     end_date = today_begin_date - timedelta(days=(i - 1))










    for i in range(0, 31):
        # 这样就可以表示今天一天的用户增加的数量
        # 表示今天的０点０分
        # timedelta:计算两个时间差
        # 2018-06-18 0点0分0秒
        begin_date = today_begin_date - timedelta(days=i)
        # 表示昨天的０点０分
        # 2018-06-19 0点0分0秒
        end_date = today_begin_date - timedelta(days=(i - 1))

        count = User.query.filter(User.is_admin == False, User.create_time >= begin_date,
                                  User.create_time < end_date).count()
        activate_count.append(count)
        activate_time.append(begin_date.strftime("%Y-%m-%d"))

    activate_time.reverse()
    activate_count.reverse()
    data = {
        "total_count": sum_count,
        "mon_count": mon_count,
        "day_count": day_count,
        "activate_count": activate_count,
        "activate_time": activate_time
    }
    return render_template("admin/user_count.html", data=data)


@admin_blue.route("/index")
@user_login_data
def admin_index():
    user = g.user
    return render_template("admin/index.html", user=user.to_dict())


@admin_blue.route("/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        user_id = session.get("user_id", None)
        is_admin = session.get("is_admin", False)
        # 如果当前用户已经登陆，并且还是管理员，才能进入到管理员的界面
        if user_id and is_admin:
            return redirect(url_for("admin.admin_index"))
        return render_template("admin/login.html")

    username = request.form.get("username")
    password = request.form.get("password")
    # 如果用户名正确，还必须是管理员才能登陆
    user = User.query.filter(User.mobile == username, User.is_admin == True).first()

    if not user:
        return render_template("admin/login.html", errmsg="没有这个用户")

    if not user.check_password(password):
        return render_template("admin/login.html", errmsg="密码错误")

    session["user_id"] = user.id
    session["mobile"] = user.mobile
    session["nick_name"] = user.nick_name
    session["is_admin"] = user.is_admin

    return redirect(url_for("admin.admin_index"))
