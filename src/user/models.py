
# from datetime import datetime
# from sqlalchemy import MetaData, Table,Column, Integer,Float, String,TIMESTAMP,ForeignKey,JSON

# metadata=MetaData()

# users = Table(
#     'user',
#     metadata,
#     Column("id",Integer,primary_key=True),
#     Column("login", String, nullable=False),
#     Column("pasword", String, nullable=False),
#     Column("registered_at", TIMESTAMP,nullable=False,default=datetime.utcnow ),
#     Column("check", Float,nullable=False,default=0),
# )

# from datetime import datetime
# from sqlalchemy import TIMESTAMP, Column, Float, Integer, String
# from src.database import Base
# class User(Base):
#     __tablename__ = "user"

#     id=Column(Integer,primary_key=True)
#     login= Column( String, nullable=False)
#     pasword=Column( String, nullable=False)
#     registered_at= Column( TIMESTAMP,nullable=False,default=datetime.utcnow )
#     check= Column( Float,nullable=False,default=0)