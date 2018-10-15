from flask import render_template,current_app
from flask import request,jsonify
from flask import session

from info.models import User, News, Category
from info.utils.response_code import RET
from . import index_blue









@index_blue.route("/news_list")
def news_list():
    # 第二个参数表示默认值
    # 获取到分类id
    cid = request.args.get("cid",1)
    # 表示当前在哪一个页面
    page = request.args.get("page",1)
    # 每页表示多少条数据,默认是10条数据
    per_page = request.args.get("per_page",10)

    try:
        cid = int(cid)
        page = int(page)
        per_page = int(per_page)
    except Exception as e:
        current_app.logger.error(e)
        return jsonify(errno = RET.PARAMERR,errmsg = "参数错误")

    # filter = [News.status == 0]
    filter =  []
    if cid != 1:
        filter.append(News.category_id == cid)
    # paginate:表示分页
    # 第一个参数表示,表示哪个页面
    # 第二个参数表示每页有多少条数据
    # 第三个参数是否有错误输出
    # if cid == 1:
    #    paginate =News.query.order_by(News.create_time.desc()).paginate(page,per_page,False)
    # else:
    paginate = News.query.filter(*filter).order_by(News.create_time.desc()).paginate(page,per_page,False)
    # 当前要展示页面所有的数据
    items = paginate.items
    # 获取到当前页面
    current_page = paginate.page
    # 获取到总页数
    total_page = paginate.pages


    news_dict_list = []
    for news_item in items:
        news_dict_list.append(news_item.to_dict())




    data = {
        "current_page":current_page,
        "total_page":total_page,
        "news_dict_li":news_dict_list,
        "cid":cid
    }

    return jsonify(errno = RET.OK,errmsg = "成功",data = data)




# 请求图标,图标的路径是固定的写法
@index_blue.route("/favicon.ico")
def favicon():
    # 这个图标是固定的写法 ,注意:千万不要忘记了后缀.ico
    return current_app.send_static_file("news/favicon.ico")


@index_blue.route("/")
def index():
    # 从seesion当中获取到user_id,因为登陆成功之后,把user_id存储到session里面
    user_id = session.get("user_id")
    user = None
    if user_id:
        # 如果能够从session当中获取user_id,说明用户已经登陆
       user = User.query.get(user_id)

    """
    查询右边热门新闻的数据
    """
    news_model = News.query.order_by(News.clicks.desc()).limit(10)

    news_clicks = []

    for news in  news_model:
        news_clicks.append(news.to_dict())


    """
    首页最上面的分类数据
    """
    category_model = Category.query.all()

    categorys = []

    for category in category_model:
        categorys.append(category.to_dict())



    data = {
        # 为了前端工程展示方便,所以需要把user对象里面的数据转换成字典
        "user_info":user.to_dict() if user else None,
        "click_news_list":news_clicks,
        "categories":categorys
    }

    return render_template("news/index.html",data = data)