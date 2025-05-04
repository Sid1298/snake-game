from datetime import datetime
from sqlalchemy import create_engine, desc, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///db.sql", echo=True)
Base = declarative_base()

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    score = Column(Integer, nullable=False, default=0)
    playdate = Column(DateTime, nullable=False, default=datetime.now())


Base.metadata.create_all(engine)


def add_new_score(user_score):
    global engine
    Session = sessionmaker(engine)
    session = Session()
    session.add(user_score)
    session.commit()
    session.close()
    

def get_top_5_scores():
    global engine
    Session = sessionmaker(engine)
    session = Session()
    users = session.query(User).order_by(desc(User.score)).order_by(desc(User.playdate)).limit(5).all()
    return users
