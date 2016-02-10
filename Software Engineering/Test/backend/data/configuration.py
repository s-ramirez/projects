from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('mysql://root:@localhost:3306/testDB', echo=False)
Session = sessionmaker(bind=engine)
session = Session()
