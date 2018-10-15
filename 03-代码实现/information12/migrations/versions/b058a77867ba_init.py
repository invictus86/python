"""init

Revision ID: b058a77867ba
Revises: 
Create Date: 2018-06-20 13:58:22.659950

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b058a77867ba'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('info_category',
    sa.Column('create_time', sa.DateTime(), nullable=True),
    sa.Column('update_time', sa.DateTime(), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('info_user',
    sa.Column('create_time', sa.DateTime(), nullable=True),
    sa.Column('update_time', sa.DateTime(), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('nick_name', sa.String(length=32), nullable=False),
    sa.Column('password_hash', sa.String(length=128), nullable=False),
    sa.Column('mobile', sa.String(length=11), nullable=False),
    sa.Column('avatar_url', sa.String(length=256), nullable=True),
    sa.Column('last_login', sa.DateTime(), nullable=True),
    sa.Column('is_admin', sa.Boolean(), nullable=True),
    sa.Column('signature', sa.String(length=512), nullable=True),
    sa.Column('gender', sa.Enum('MAN', 'WOMAN'), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('mobile'),
    sa.UniqueConstraint('nick_name')
    )
    op.create_table('info_news',
    sa.Column('create_time', sa.DateTime(), nullable=True),
    sa.Column('update_time', sa.DateTime(), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(length=256), nullable=False),
    sa.Column('source', sa.String(length=64), nullable=False),
    sa.Column('digest', sa.String(length=512), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('clicks', sa.Integer(), nullable=True),
    sa.Column('index_image_url', sa.String(length=256), nullable=True),
    sa.Column('category_id', sa.Integer(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('status', sa.Integer(), nullable=True),
    sa.Column('reason', sa.String(length=256), nullable=True),
    sa.ForeignKeyConstraint(['category_id'], ['info_category.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['info_user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('info_user_fans',
    sa.Column('follower_id', sa.Integer(), nullable=False),
    sa.Column('followed_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['followed_id'], ['info_user.id'], ),
    sa.ForeignKeyConstraint(['follower_id'], ['info_user.id'], ),
    sa.PrimaryKeyConstraint('follower_id', 'followed_id')
    )
    op.create_table('info_comment',
    sa.Column('create_time', sa.DateTime(), nullable=True),
    sa.Column('update_time', sa.DateTime(), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('news_id', sa.Integer(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('parent_id', sa.Integer(), nullable=True),
    sa.Column('like_count', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['news_id'], ['info_news.id'], ),
    sa.ForeignKeyConstraint(['parent_id'], ['info_comment.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['info_user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('info_user_collection',
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('news_id', sa.Integer(), nullable=False),
    sa.Column('create_time', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['news_id'], ['info_news.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['info_user.id'], ),
    sa.PrimaryKeyConstraint('user_id', 'news_id')
    )
    op.create_table('info_comment_like',
    sa.Column('create_time', sa.DateTime(), nullable=True),
    sa.Column('update_time', sa.DateTime(), nullable=True),
    sa.Column('comment_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['comment_id'], ['info_comment.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['info_user.id'], ),
    sa.PrimaryKeyConstraint('comment_id', 'user_id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('info_comment_like')
    op.drop_table('info_user_collection')
    op.drop_table('info_comment')
    op.drop_table('info_user_fans')
    op.drop_table('info_news')
    op.drop_table('info_user')
    op.drop_table('info_category')
    # ### end Alembic commands ###
