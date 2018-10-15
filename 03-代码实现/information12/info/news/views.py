from flask import current_app
from flask import g
from flask import request
from flask import session

from info import db
from info.models import User, News, Comment, CommentLike
from info.utils.common import user_login_data
from info.utils.response_code import RET
from . import news_blue
from flask import render_template,jsonify

"""
关注与取消关注
"""

@news_blue.route("/followed_user",methods = ["POST"])
@user_login_data
def followed_user():
    # 因为当前的接口是实现关注和取消关注,所有必须得登陆才能关注
    user = g.user
    if not user:
        return jsonify(errno = RET.SESSIONERR,errmsg = "请登录")

    # 我要关注另外一个用户的id
    user_id = request.json.get("user_id")
    # 表示当前是需要关注还是取消关注
    action = request.json.get("action")

    if not all([user_id,action]):
        return jsonify(errno = RET.PARAMERR,errmsg = "参数错误")

    # 查询出来,我想关注的另外一个用户
    other = User.query.get(user_id)

    if action == "follow":
        # 表示想关注当前的用户
        # 如果当前我想关注的这个人,不在我的关注列表当中,那么就可以直接关注,如果在,那么就不能关注
        if other not in user.followed:
            user.followed.append(other)
        else:
            return jsonify(errno = RET.DATAERR,errmsg = "已经被关注了")
    else:
        # 取消关注
        # 如果当前需要关注的这个人,已经在我的关注人列表当中,直接取消关注
        if other in user.followed:
            user.followed.remove(other)
        else:
            return jsonify(errno=RET.DATAERR, errmsg="没有关注")

    db.session.commit()
    return jsonify(errno = RET.OK,errmsg = "OK")



@news_blue.route("/comment_like",methods = ["POST"])
@user_login_data
def comment_like():
    user = g.user
    if not user:
        return jsonify(errno=RET.SESSIONERR, errmsg="请登陆")

    comment_id = request.json.get("comment_id")
    news_id = request.json.get("news_id")
    # 获取到用户点赞的动作，那么点赞，那么取消点赞
    action = request.json.get("action")

    comment = Comment.query.get(comment_id)
    """
    实现评论点赞：
    １　当前点赞的用户是谁　user.id
    2  需要知道当前点的哪条评论,comment_id　　　　　　　　　　　　　　　　　　　  　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
    """
    if action == "add":
        # 点赞
        comment_like = CommentLike.query.filter(CommentLike.comment_id == comment_id,CommentLike.user_id == user.id).first()
        if not comment_like:
            # 如果之前没有点赞，可以进行点赞，如果之前已经点赞，那么在点击就是取消
            comment_like = CommentLike()
            comment_like.comment_id = comment_id
            comment_like.user_id = user.id
            db.session.add(comment_like)
            comment.like_count += 1
    else:
        comment_like = CommentLike.query.filter(CommentLike.comment_id == comment_id,
                                                CommentLike.user_id == user.id).first()
        if comment_like:
            db.session.delete(comment_like)
            comment.like_count -= 1


    db.session.commit()
    return jsonify(errno = RET.OK,errmsg = "点赞成功")



@news_blue.route("/news_comment" ,methods = ["POST"])
@user_login_data
def news_comment():
    user = g.user
    if not user:
        return jsonify(errno = RET.SESSIONERR,errmsg = "请登陆")

    news_id = request.json.get("news_id")
    comment_str = request.json.get("comment")
    # 父类评论，一般表示神回复
    parent_id = request.json.get("parent_id")

    news = News.query.get(news_id)

    comment = Comment()
    comment.user_id = user.id
    comment.news_id = news.id
    comment.content = comment_str
    if parent_id:
        comment.parent_id = parent_id

    db.session.add(comment)
    db.session.commit()

    return jsonify(errno = RET.OK,errmsg = "评论成功",data = comment.to_dict())








@news_blue.route("/news_collect" ,methods = ["POST"])
@user_login_data
def news_collect():

    user = g.user
    if not user:
        return jsonify(errno = RET.SESSIONERR,errmsg = "请登陆")

    # 表示获取到新闻id
    news_id = request.json.get("news_id")
    # 获取到当前收藏的动作,收藏或者取消收藏
    action = request.json.get("action")
    # 查询需要进行收藏的新闻
    news = News.query.get(news_id)

    if not news:
        return jsonify(errno = RET.NODATA,errmsg = "没有新闻数据")

    if action == "collect":
        # 收藏
        user.collection_news.append(news)
    else:
        # 取消收藏
        user.collection_news.remove(news)

    try:
        db.session.commit()
    except Exception as e:
        current_app.logger.error(e)
        db.session.rollback()

    return jsonify(errno = RET.OK,errmsg = "收藏成功")




@news_blue.route("/<int:news_id>")
@user_login_data
def news_detail(news_id):
    # # 从seesion当中获取到user_id,因为登陆成功之后,把user_id存储到session里面
    # user_id = session.get("user_id")
    # user = None
    # if user_id:
    #     # 如果能够从session当中获取user_id,说明用户已经登陆
    #     user = User.query.get(user_id)
    user = g.user
    """
       查询右边热门新闻的数据
       """
    news_model = News.query.order_by(News.clicks.desc()).limit(10)

    news_clicks = []

    for news in news_model:
        news_clicks.append(news.to_dict())

    """
    根据新闻id查询当前新闻详情里面的数据
    """
    news = News.query.get(news_id)
    news.clicks += 1

    """
    收藏新闻
    """
    # 默认情况下,所有的新闻都是没有被收藏,所以设置为false
    is_collected = False

    if user:
        # 判断当前查询出来的新闻,如果在用户收藏的列表当中,说明当前新闻已经被收藏,所以设置为ture
        if news in user.collection_news:
            is_collected = True

    """
    获取到新闻详情的评论信息
    """
    comments = Comment.query.filter(Comment.news_id == news_id).all()
    comment_likes = []
    comment_like_ids = []
    if user:
        # 查询用户点赞了哪些评论
        comment_likes = CommentLike.query.filter(CommentLike.user_id == user.id).all()
        # 取出来所有点赞的评论id
        comment_like_ids = [comment_like.comment_id for comment_like in comment_likes]

    commnets_list = []
    for item in  comments:
        item_dict = item.to_dict()
        # 表示默认情况下，所有的评论都没有被点赞
        item_dict["is_like"] = False
        if item.id in comment_like_ids:
          item_dict["is_like"] = True

        commnets_list.append(item_dict)

    # 当前登陆用户没有关注新闻的作者,默认情况下,肯定是没有关注,所以设置为false
    is_followed = False
    # 如果当前登陆用户需要关注新闻作者的话,那么必须首先得有新闻作者
    # 用户登陆之后,才能进行关注
    if news.user and user:
        # 如果当前新闻的作者,在登陆用户的关注人列表当中,就可以说明当前登陆作者是新闻作者的粉丝
        if news.user in user.followed:
            is_followed = True


    data = {
        "user_info": user.to_dict() if user else None,
        "click_news_list": news_clicks,
        "news":news.to_dict(),
        "comments":commnets_list,
        "is_followed":is_followed,
        "is_collected":is_collected
    }
    return render_template("news/detail.html",data = data)